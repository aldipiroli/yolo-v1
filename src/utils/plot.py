import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from src.utils.utils import get_box_coord
import numpy as np


def plot_boxes(box, H=448, W=448, S=7, fig=None, ax=None, color="red", plot_corners=False):
    """ 
        Debug function for plotting network outputs
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    img = mpimg.imread("/home/aldi/workspace/yolo-v1/img/cat.png")[:H, :W, :]
    ax.imshow(img)

    for i in range(box.shape[0]):
        for j in range(box.shape[1]):
            if box[i, j, 4:].any() != 0:
                x_ = H / S * i
                y_ = W / S * j

                # Plot Points:
                points = get_box_coord(box[i, j, :])
                points += [x_, y_]
                if plot_corners:
                    for i in range(points.shape[0]):
                        ax.plot(points[i, 0], points[i, 1], color=color, marker="o")
                        ax.text(points[i, 0], points[i, 1], str(i))

                # Plot Box:
                pt_idx = [[0, 1], [1, 2], [2, 3], [3, 0]]
                for idx in range(len(pt_idx)):
                    i, j = pt_idx[idx][0], pt_idx[idx][1]
                    ax.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], color=color)

    return fig, ax


def plot_voc2007_boxes(
    img,
    annotations,
    fig=None,
    ax=None,
    color="red",
):
    """
    Plots the labels in the standartd format of the annotation xml files:
        [x_min ,y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    ax.imshow(img)

    for ann in annotations:
        x_min, y_min = ann[0], ann[1]
        x_max, y_max = ann[2], ann[3]

        p1 = [x_min, y_min]
        p2 = [x_max, y_min]
        p3 = [x_max, y_max]
        p4 = [x_min, y_max]

        p = np.array([p1, p2, p3, p4, p1])
        ax.plot(p[:, 0], p[:, 1], color=color, marker="o")

    plt.show()


def plot_voc2007_labels(img, annotations, fig=None, ax=None, color="red"):
    """ Plots the labels in the yolo format:
            center, height, weight
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    ax.imshow(img)

    H = img.shape[0]
    S = annotations.shape[0]

    STEP = H / S
    for i in range(S):
        for j in range(S):
            STEP_X = i * STEP
            STEP_Y = j * STEP
            ann = annotations[i, j, :]
            if ann[4:].any() != 0:
                x = ann[0]
                y = ann[1]
                w = ann[2] / 2
                h = ann[3] / 2

                p1 = [x - w + STEP_X, y - h + STEP_Y]
                p2 = [x + w + STEP_X, y - h + STEP_Y]
                p3 = [x + w + STEP_X, y + h + STEP_Y]
                p4 = [x - w + STEP_X, y + h + STEP_Y]

                p = np.array([p1, p2, p3, p4, p1])
                ax.plot(p[:, 0], p[:, 1], color=color, marker="o")

    plt.show()
