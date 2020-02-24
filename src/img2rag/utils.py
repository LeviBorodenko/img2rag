import numpy as np

__author__ = "Levi Borodenko"
__copyright__ = "Levi Borodenko"
__license__ = "mit"


def attributes2array(data: dict) -> np.ndarray:
    """Takes a dict and turns its items into
    an array

    [a, b, c, ...]


    Arguments:
        data {dict} -- Dict containing int, float or list items.

    Returns:
        np.ndarray -- "item1, item2, listsublitem1, ..."
    """

    attributes = []

    for key, value in data.items():

        if key == "labels":
            pass
        elif isinstance(value, (int, float)):
            attributes.append(value)
        elif isinstance(value, (list, np.ndarray)):
            for item in value:
                attributes.append(item)

    array = np.asarray(attributes, dtype=np.float32)

    return array
