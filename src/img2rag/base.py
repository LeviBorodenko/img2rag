import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from cached_property import cached_property
from matplotlib import pyplot as plt
from skimage import color, data, measure, segmentation
from skimage.future import graph

from img2rag.utils import attributes2array

__author__ = "Levi Borodenko"
__copyright__ = "Levi Borodenko"
__license__ = "mit"


class RAGimage(object):
    """Takes an image and allows you to calculate
    a Region Adjacency Graph based of k-means and
    a subsequent threshold cut.

    Arguments:
        image_tensor (np.ndarray): Height x width x channel
            array encoding the image. Can drop channel dimension
            for gray scale images.

    Keyword Arguments:
        rag_threshold (float): threshold value for fusing
            regions. If regions have color difference less than
            rag_threshold, they will be fused into one region.
            (default: {25})
    """

    def __init__(self, image_tensor: np.ndarray, rag_threshold: float = 100):
        super(RAGimage, self).__init__()

        # convert to numpy array if necessary
        # and check if number of dimensions
        # makes sense for an image.
        image_tensor = np.asarray(image_tensor)
        self.shape = image_tensor.shape

        dims = len(self.shape)

        if dims > 3 or dims == 1:
            raise ValueError(
                f"Image should have 2 or 3 dimensions. \
                               got shape: {self.shape}"
            )
        elif dims == 3:
            self.channels = self.shape[-1]

        else:
            self.channels = None

        self.height = image_tensor.shape[0]
        self.width = image_tensor.shape[1]

        self.image_tensor = image_tensor

        self.rag_threshold = rag_threshold

    def get_segementation_labels(self, **kwargs) -> np.ndarray:
        """Performs initial segmentation of the image using
        k-means in a 5D color-location space.

        Arguments:
            **kwargs: Arguments passed to the
                k-means algorithm.

        Returns:
            np.ndarray: label array
        """

        label = segmentation.felzenszwalb(self.image_tensor, **kwargs)

        return label

    @cached_property
    def segmentation_labels(self) -> np.ndarray:
        """Array that has labels for each
        corresponding pixel.

        Decorators:
            cached_property

        Returns:
            np.ndarray: (Height, Width) shaped label array.
        """

        return self.get_segementation_labels()

    def get_rag_labels_signal(self):
        """Main working horse.

        Calculates the RAG, its corresponding labels and
        graph signal.
        """

        # first we create an initial RAG based on the k-means
        # segmentation
        rag = graph.rag_mean_color(self.image_tensor, self.segmentation_labels)

        labels = graph.cut_threshold(self.segmentation_labels, rag, thresh=20)

        # create rag from these new labels
        rag = graph.rag_mean_color(self.image_tensor, labels)

        # add 1 so labels start at 1
        labels = labels + 1

        # We will add a few more node attributes
        properties = measure.regionprops(labels)

        # we will create the graph signal now as well
        # so we only have to iterate through all the regions once.
        X = []

        # properties now contains a dict of various things that
        # we can calculate for each region. It is indexed by label.
        for region in rag.nodes:
            idx = region

            # get corresponding properties
            props = properties[idx]

            # get centeroid and normalise it
            centroid_x, centroid_y = props.centroid
            centroid_x = centroid_x / self.width
            centroid_y = centroid_y / self.height

            # get orientation of region
            orientation = props.orientation

            # update node
            rag.nodes[idx]["centroid"] = [centroid_x, centroid_y]
            rag.nodes[idx]["orientation"] = orientation

            # turn all the node attributes into an ordered array
            # and append it as a row to the graph signal
            X.append(attributes2array(rag.nodes[idx]))

        # stack X rows to create one array
        X = np.stack(X)

        return (rag, labels, X)

    @cached_property
    def rag_labels_signal(self):
        return self.get_rag_labels_signal()

    @cached_property
    def rag(self) -> nx.DiGraph:
        """Returns the RAG as a networkx DiGraph.

        Decorators:
            cached_property
        """
        rag, labels, X = self.rag_labels_signal
        return rag

    @cached_property
    def labels(self) -> np.ndarray:
        """Lables of final segmentation.
        """
        rag, labels, X = self.rag_labels_signal
        return labels

    @cached_property
    def signal(self) -> np.ndarray:
        """Graph signal corresponding to the RAG.
        """
        rag, labels, X = self.rag_labels_signal
        return X

    @cached_property
    def adjacency(self) -> np.ndarray:
        """Adjacency matrix of RAG.
        """

        return nx.to_numpy_matrix(self.rag, weight=None)

    @cached_property
    def edge_list(self) -> list:
        """Edge list of RAG.
        """
        return list(nx.generate_edgelist(self.rag, data=False))

    def show_result(self):
        """Shows the initial and final segregation of
        the image.
        """
        segmentation_img = color.label2rgb(
            self.segmentation_labels, self.image_tensor, kind="avg"
        )

        final_img = color.label2rgb(self.labels, self.image_tensor, kind="avg")

        fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))

        ax[0].imshow(segmentation_img)
        ax[1].imshow(final_img)

        for a in ax:
            a.axis("off")

        plt.tight_layout()
        plt.show()


class Shard(object):
    """Helper class that represents a shard of an
    (image_tensor, label) tf.data.Dataset.

    Arguments:
        dataset (tf.data.Dataset): (image, label) dataset.
            shard_num (int): Number of shards
            shard_mod (int): What shard to represent

    Keyword Arguments:
        name (str): prefix name of save file (default: {"DATA"})
        save_folder {Path}: folder in which to save the
            processed rag_labels_signal (default: {"./"})
    """

    def __init__(
        self,
        dataset: tf.data.Dataset,
        shard_num: int,
        shard_mod: int,
        name: str = "DATA",
        save_folder: Path = "./",
    ):
        super(Shard, self).__init__()

        self.data = dataset
        self.shard = dataset.shard(shard_num, shard_mod)

        filename = f"{name}-{shard_num}-{shard_mod}"
        self.save_file = Path(save_folder) / filename

        self.count = 0

    def process_one(self, data_item):

        # data is the image and label the integer label
        image_tensor, label = data_item

        # create image instance
        img = Image(image_tensor)

        result = ({"edge_list": img.edge_list, "signal": img.signal}, label)

        self.count += 1

        print(f"Processed {self.count} images.")

        return result

    def process_all(self):

        result_all = []
        for data_item in self.shard:

            try:
                result_one = self.process_one(data_item)
                result_all.append(result_one)
            except KeyError:
                print("Empty graph!")
                self.count -= 1

        print(f"Done! Created {len(result_all)} samples.")
        print(f"Saving to {self.save_file}")

        with open(self.save_file, "wb") as f:
            pickle.dump(result_all, f)


class ShardFuser(object):
    """docstring for ShardFuser"""

    def __init__(self, data_folder: Path, save_file: Path):
        super(ShardFuser, self).__init__()
        self.data_folder = Path(data_folder)
        self.save_file = Path(save_file)

    def fuse(self):
        result = []

        for file in self.data_folder.iterdir():

            with open(file, "rb") as pickled:

                data_list = pickle.load(pickled)

                for data_item in data_list:

                    result.append(data_item)

        with open(self.save_file, "wb") as save_file:
            pickle.dump(result, save_file)


if __name__ == "__main__":
    caltech = tfds.load("caltech101", as_supervised=True)
    train = caltech["train"]

    dataset = train.shuffle(1000).take(1)

    for img, label in dataset:
        image = Image(img, rag_threshold=10)
        image.show_result()
