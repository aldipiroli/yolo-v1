import torch
import numpy as np
import random
from random import randrange
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches


def plot_boxes(gt, pr,  H, W, S):
    fig, ax = plt.subplots()
    img = mpimg.imread("../img/cat.png")[:H, :W]
    plt.imshow(img)

    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i, j, 4] != -1:
                x_ = H / S * i
                y_ = W / S * j

                rect = patches.Rectangle(
                    (x_ + gt[i, j, 0], y_ + gt[i, j, 1]), gt[i, j, 2], gt[i, j, 3], linewidth=2, edgecolor="r", facecolor="none"
                )
                ax.add_patch(rect)

            if pr[i, j, 4] != -1:
                x_ = H / S * i
                y_ = W / S * j

                rect = patches.Rectangle(
                    (x_ + pr[i, j, 0], y_ + pr[i, j, 1]), pr[i, j, 2], pr[i, j, 3], linewidth=2, edgecolor="lime", facecolor="none"
                )
                ax.add_patch(rect)
            
    plt.show()


def make_gt(S, C, W, H):
    gt = np.zeros((S, S, 5))
    for i in range(S):
        for j in range(S):
            x = random.uniform(0.0, 1.0)
            y = random.uniform(0.0, 1.0)
            w = random.uniform(0.0, H / S)
            h = random.uniform(0.0, W / S)
            rand_pr = random.uniform(0.0, 1.0)
            if rand_pr > 0.9:
                rnd_cls = randrange(C)
            else:
                rnd_cls = -1
            gt[i, j] = [x, y, w, h, rnd_cls]

    return gt

def make_pr(gt, B, err=5):
    S = gt.shape[0]
    pr = np.zeros((S, S, B*5))
    print(pr.shape)
    for i in range(S):
        for j in range(S):
            if gt[i, j, 4] != -1:
                rand_pr = random.uniform(0.0, 1.0)
                if rand_pr > 0.3:
                    pr[i,j, :5] = gt[i, j, :] + random.uniform(0.0, err)
                    pr[i,j, 5:] = gt[i, j, :] + random.uniform(0.0, err)
    return pr



def yolo_loss(gt, pr):
    """
    gt: SxS [x,y,h,w,class]
    pr: SxSx[B*5 + C]
    """

    lamb_coord = 5
    lamb_noobj = .5

    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            

    

    print("end")

if __name__ == "__main__":
    W, H = 448, 448
    S, B, C = 7, 2, 20
    pr = []
    gt = make_gt(S, C, W, H)
    pr = make_pr(gt, B)

    plot_boxes(gt,pr, H, W, S)
    # yolo_loss(gt, pr)
