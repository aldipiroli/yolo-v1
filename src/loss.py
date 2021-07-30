import torch
import numpy as np
import random
import math
from random import randrange
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches



def IoU(gt_box, pr_box):
    """
    IoU = Area Overlap / Area Total
    0------------------1
    |                  |
    |                  |
    |                  |
    3------------------2
    """
    print("gt_box\n", gt_box)
    print("pr_box\n", pr_box)
    # Compute Intersection:
    x_1 = min(gt_box[1][0], pr_box[1][0])
    x_2 = max(gt_box[0][0], pr_box[0][0])
    dx = x_1 - x_2

    y_1 = min(gt_box[3][0], pr_box[3][0])
    y_2 = max(gt_box[0][1], pr_box[0][1])
    dy = y_1 - y_2

    inter = dx * dy

    if inter < 0:
        return 0

    # Compute Union:
    gt_dx = np.absolute(gt_box[1][0] - gt_box[0][0])
    gt_dy = np.absolute(gt_box[3][1] - gt_box[0][1])

    pr_dx = np.absolute(pr_box[1][0] - pr_box[0][0])
    pr_dy = np.absolute(pr_box[3][1] - pr_box[0][1])

    union = gt_dx * gt_dy + pr_dx * pr_dy - inter

    return inter / union


def compute_loss(gt, pr, H, W, S):
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            x_ = H / S * i
            y_ = W / S * j
            if gt[i, j, 4] != -1:
                gt_box = get_box_coord(gt[i, j]) + [x_, y_]
                pr_box1 = get_box_coord(pr[i, j, :5]) + [x_, y_]
                pr_box2 = get_box_coord(pr[i, j, 5:]) + [x_, y_]
                iou_box_1 = IoU(gt_box, pr_box1)
                iou_box_2 = IoU(gt_box, pr_box2)

                print("IoU: ", iou_box_1, iou_box_2)


if __name__ == "__main__":
    print("OK")
    # W, H = 448, 448
    # S, B, C = 7, 2, 20
    # pr = []
    # gt = make_gt(S, C, W, H, fake=False)
    # pr = make_pr(gt, B, fake=False)

    # fig, ax = plot_boxes(gt, color="red")
    # fig, ax = plot_boxes(pr[:, :, :5], color="lime", fig=fig, ax=ax)
    # fig, ax = plot_boxes(pr[:, :, 5:], color="orange", fig=fig, ax=ax)
    # plt.show()
    # # yolo_loss(gt, pr)
    # compute_loss(gt, pr, H, W, S)
