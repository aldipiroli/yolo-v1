import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

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
            if ann[4] != 0:
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

    return fig, ax 
