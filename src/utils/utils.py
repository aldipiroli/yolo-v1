import numpy as np


def get_box_coord(box):
    """
    Input:
    +------------------+
    |                  |
    H                  |
    |                  |
    (xy)-------W-------+

    Output:
    A------------------B
    |                  |
    |                  |
    |                  |
    C------------------D
    """
    x = box[0]
    y = box[1]
    W = box[2]
    H = box[3]

    A = [x, y - H]
    B = [x + W, y - H]
    C = [x, y]
    D = [x + W, y]

    return np.array((A, B, D, C))
