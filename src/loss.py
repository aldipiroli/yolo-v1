import torch
import numpy as np
import random
import math
from random import randrange
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

from src.utils.utils import get_box_coord


def IoU(gt_box, pr_box):
    """
    IoU = Area Overlap / Area Total
    0------------------1
    |                  |
    |                  |
    |                  |
    3------------------2
    """
    # Compute Intersection:
    x_list = sorted([gt_box[0][0], gt_box[1][0], pr_box[0][0], pr_box[1][0]])
    dx = np.absolute(x_list[1] - x_list[2])

    y_list = sorted([gt_box[0][1], gt_box[3][1], pr_box[0][1], pr_box[3][1]])
    dy = np.absolute(y_list[1] - y_list[2])

    inter = dx * dy

    if inter < 0:
        return 0

    # Compute Union:
    gt_dx = np.absolute(gt_box[1][0] - gt_box[0][0])
    gt_dy = np.absolute(gt_box[3][1] - gt_box[0][1])

    pr_dx = np.absolute(pr_box[1][0] - pr_box[0][0])
    pr_dy = np.absolute(pr_box[3][1] - pr_box[0][1])

    print("=" * 50)
    union = gt_dx * gt_dy + pr_dx * pr_dy - inter

    print("GT BBox: \n", gt_box)
    print("PR BBox: \n", pr_box)

    print("GT dx %.2f, dy %.2f" % (gt_dx, gt_dy))
    print("PR dx %.2f, dy %.2f" % (pr_dx, pr_dy))

    print("Inter %.2f, dx %.2f, dy %.2f" % (inter, dx, dy))
    print("x_list ", x_list, "y_list", y_list)

    print("IoU: %.2f, Int: %.2f, Union: %.2f " % (inter / union, inter, union))
    print("=" * 50)
    return inter / union


def compute_loss(gt, pr, H=448, W=448, S=7):
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            x_ = H / S * i
            y_ = W / S * j
            if gt[i, j, 4] != -1:
                gt_box = get_box_coord(gt[i, j]) + [x_, y_]
                pr_box1 = get_box_coord(pr[i, j, :5]) + [x_, y_]
                pr_box2 = get_box_coord(pr[i, j, 5:]) + [x_, y_]
                iou_box_1 = IoU(gt_box, pr_box1)
                # iou_box_2 = IoU(gt_box, pr_box2)

                # pr_box, pr_idx = (pr_box1, 1) if iou_box_1 >= iou_box_2 else (pr_box2, 2)
                # print("*"*20)
                # print("GT BBox: \n", gt_box)
                # print("PR BBox 1: \n", pr_box1)
                # print("PR BBox 2: \n", pr_box2)
                # print("IoU Box1: %.2f, Box2: %.2f, Taken %d" % (iou_box_1, iou_box_2, pr_idx))
