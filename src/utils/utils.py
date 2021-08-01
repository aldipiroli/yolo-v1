import numpy as np


def get_box_coord(box):
    """
    Input:
    box[0], box[1]: center of the cell (cell reference [0,1])
    box[2], box[3]: height, width of the cell (image reference [H,W])

    Output:
    A-------------------
    |                  |
    |                  |
    |                  |
    -------------------B
    """
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]

    A = [x - w / 2, y - h / 2]
    B = [x + w / 2, y + h / 2]

    return np.array((A, B))
