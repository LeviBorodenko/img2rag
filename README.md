## img2rag

Convert any image into its[Region Adjacency Graph](https://ieeexplore.ieee.org/document/841950) which can be used for either image segmentation or to create a graph embedding of the image.

<hr>

### Installation

Simply run `pip install img2rag`

### What is does

Given an image, we segement it into morphological regions using first[Felzenszwalb segmentation](http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf) followed by a[threshold - cut](https://ieeexplore.ieee.org/document/841950). Given these segmeneted regions, we now construct the following graph:

1. Each node corresponds to a segmented region.
2. We connect two regions if they are adjacent.

This is the so - called region - adjacency graph. Furthermore, we add the following node - attributes to each region:

1. Location of the region centeriod
2. Orientation of the region
3. Mean and total color of the region
4. Size in px

The edges contain the mean - color - difference between the two regions

# How to use

Simply import the `RAGimage` class and initiate with any image. Then use the build in methods to access various properties.


```python
from img2rag import RAGimage

# We assume the image is given as a numpy array or tf.Tensor with either 2 or 3 dimensions
# where the third dimension is the optional channel dimension.
img_tensor = [...]

# initiate RAGiamge instance
image_rag = RAGimage(img_tensor)

# RAG as a networkx attributed DiGraph
image_rag.rag

# Scikit style labels of the image segementation
image_rag.labels

# Adjacency matric of the RAG
image_rag.adjacency

# Graph feature matrix of the RAG
# (Nodes x Node-Features)
image_rag.signal
```
