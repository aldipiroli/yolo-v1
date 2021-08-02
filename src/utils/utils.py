import numpy as np
from skimage import io, transform
import matplotlib.image as mpimg


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


def resize_data(image_file, annotations, output_size=(448, 448)):
    img = mpimg.imread(image_file)
    h, w = img.shape[0], img.shape[1]

    new_h, new_w = output_size
    img = transform.resize(img, (new_h, new_w))

    ann_ratio = np.array([new_w / w, new_h / h])

    for ann in annotations:
        ann[0] = np.clip(ann[0] * ann_ratio[0], 0, new_w)
        ann[1] = np.clip(ann[1] * ann_ratio[1], 0, new_h)
        ann[2] = np.clip(ann[2] * ann_ratio[0], 0, new_w)
        ann[3] = np.clip(ann[3] * ann_ratio[1], 0, new_w)

    return img, annotations
