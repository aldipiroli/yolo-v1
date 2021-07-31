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

    union = gt_dx * gt_dy + pr_dx * pr_dy - inter

    return inter / union


def compute_loss(gt, pr, H=448, W=448, S=7):
    lamb_obj = 5
    lamb_noobj = 0.5
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            x_ = H / S * i
            y_ = W / S * j
            if gt[i, j, 4:].any() != 0:
                loss = 0
                gt_box = get_box_coord(gt[i, j, :5]) + [x_, y_]
                pr_box1 = get_box_coord(pr[i, j, :4]) + [x_, y_]
                pr_box2 = get_box_coord(pr[i, j, 5:10]) + [x_, y_]
                print("pr[i, j, 5:10]: ", pr[i, j])
                input("...")
                iou_box_1 = IoU(gt_box, pr_box1)
                iou_box_2 = IoU(gt_box, pr_box2)

                box_idx = 1 if iou_box_1 >= iou_box_2 else 2
                print("-" * 50)
                print("IoU Box1: %.2f, Box2: %.2f, Taken %d" % (iou_box_1, iou_box_2, box_idx))

                gt_ = gt[i, j]
                if box_idx == 1:
                    pr_ = pr[i, j, :5]
                else:
                    pr_ = pr[i, j, 5:]

                # position
                loss += lamb_obj * (np.square(gt_[0] - pr_[0]) + np.square(gt_[1] - pr_[1]))
                # dimensions
                loss += lamb_obj * (np.square(np.sqrt(gt_[2]) - np.sqrt(pr_[2])) + np.square(np.sqrt(gt_[3]) - np.sqrt(pr_[3])))
