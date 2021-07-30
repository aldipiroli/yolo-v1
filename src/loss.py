import torch
import numpy as np
import random
from random import randrange
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches


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
    H = box[3]
    W = box[2]

    A = [x, y - H]
    B = [x + W, y - H]
    C = [x, y]
    D = [x + W, y]

    return np.array((A, B, D, C))

def plot_boxes(box, H, W, S, fig=None, ax=None, color="red"):
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    img = mpimg.imread("../img/cat.png")[:H, :W]
    plt.imshow(img)

    # Plot Box
    for i in range(box.shape[0]):
        for j in range(box.shape[1]):
            if box[i, j, 4] != -1:
                x_ = H / S * i
                y_ = W / S * j

                # Plot Points:
                points = get_box_coord(box[i, j, :])
                points += [x_, y_]
                for i in range(points.shape[0]):
                    ax.plot(points[i, 0], points[i, 1], "ko")

                # Plot BBox:
                pt_idx = [[0,1], [1,2], [2,3], [3,0]]
                for idx in range(len(pt_idx)):
                    i,j = pt_idx[idx][0], pt_idx[idx][1]
                    ax.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], "r-")

    return fig, ax


def make_gt(S, C, W, H, fake=False):
    gt = np.zeros((S, S, 5))

    if not fake:
        for i in range(S):
            for j in range(S):
                if i > 2 and i < 5 and j > 2 and j < 5:
                    x = random.uniform(0.0, 1.0)
                    y = random.uniform(0.0, 1.0)
                    w = random.uniform(0.0, 50)
                    h = random.uniform(0.0, 50)
                    rand_pr = random.uniform(0.0, 1.0)
                    if rand_pr > 0.5:
                        rnd_cls = randrange(C)
                    else:
                        rnd_cls = -1
                    gt[i, j] = [x, y, w, h, rnd_cls]
                else:
                    gt[i, j] = [0, 0, 0, 0, -1]
    else:
        gt[:, :, 4] = -1
        gt[2,2,:] = [0.5, 0.5, 50, 100, 10]

    return gt


def make_pr(gt, B, err=5):
    S = gt.shape[0]
    pr = np.zeros((S, S, B * 5))
    print(pr.shape)
    for i in range(S):
        for j in range(S):
            if gt[i, j, 4] != -1:
                rand_pr = random.uniform(0.0, 1.0)
                if rand_pr > 0.3:
                    pr[i, j, :5] = gt[i, j, :] + random.uniform(0.0, err)
                    pr[i, j, 4] = int(pr[i, j, 4])
                    pr[i, j, 5:] = gt[i, j, :] + random.uniform(0.0, err)
                    pr[i, j, 9] = int(pr[i, j, 9])

    return pr




def IoU(gt, pr):
    """
    IoU = Area Overlap / Area Total
    +------------------+
    |                  |
    H                  |
    |                  |
    (xy)-------W-------+
    """

    fig, ax = plt.subplots()
    fig, ax = plot_boxes(gt, H, W, S, fig, ax)
    plt.show()


def yolo_loss(gt, pr):
    """
    gt: SxS [x,y,h,w,class]
    pr: SxSx[B*5 + C]
    """

    lamb_coord = 5
    lamb_noobj = 0.5

    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if i > 200 and i < 400 and j > 200 and j < 400:
                gt_box = gt[i, j]
                pr_box1 = pr[i, j, :5]
                pr_box2 = pr[i, j, 5:]
                print(gt_box)
                print(pr_box1, pr_box2)
    print("end")


if __name__ == "__main__":
    W, H = 448, 448
    S, B, C = 7, 2, 20
    pr = []
    gt = make_gt(S, C, W, H, fake=True)
    pr = make_pr(gt, B)

    # plot_boxes(gt, pr, H, W, S)
    # yolo_loss(gt, pr)
    IoU(gt, pr)
