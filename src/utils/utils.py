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
    """
    Resize an image and relative annotation to a desired output
    """
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


def is_inside_box(center_x, center_y, box):
    if center_x > box[0] and center_x < box[1]:
        if center_y > box[2] and center_y < box[3]:
            return True

    return False


def convert_coordinate_abs_rel(center_x, center_y, box):
    """Convert the coordinates from absolute value (image) to relative one of the bbox"""
    new_x = center_x - box[0]
    new_y = center_y - box[2]

    return new_x, new_y


def convert_annotation_to_label(annotations, S=7, H=448, W=448):
    label = np.zeros((S, S, 24))
    STEP = H / 7

    for i in range(S):
        for j in range(S):
            x_min, x_max = STEP * i, STEP * (i + 1)
            y_min, y_max = STEP * j, STEP * (j + 1)

            box = np.array((x_min, x_max, y_min, y_max))

            for k in range(len(annotations)):
                ann = annotations[k]
                ann_x_min, ann_y_min = ann[0], ann[1]
                ann_x_max, ann_y_max = ann[2], ann[3]
                center_x = (ann_x_max - ann_x_min) / 2 + ann_x_min
                center_y = (ann_y_max - ann_y_min) / 2 + ann_y_min

                if is_inside_box(center_x, center_y, box):
                    x, y = convert_coordinate_abs_rel(center_x, center_y, box)
                    w = ann_x_max - ann_x_min
                    h = ann_y_max - ann_y_min
                    label[i, j, :4] = [x, y, w, h]
                    label[i, j, 4:] = ann[4:]

    return label
