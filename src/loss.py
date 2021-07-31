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

    print("Inter %.2f, Union %.2f, dx %.2f, dy %.2f, gt_dx %.2f, gt_dy %.2f,  pr_dx %.2f, pr_dy %.2f " % (inter, union, dx, dy, gt_dx, gt_dy, pr_dx, pr_dy))
    return inter / union


def compute_loss(gt, pr, H=448, W=448, S=7, C=20):
    lamb_obj = 5
    lamb_noobj = 0.5

    loss = 0a
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            x_ = H / S * i
            y_ = W / S * j
            if gt[i, j, 4:].any() != 0:
                print("GT: ", gt[i, j, :4])
                print("PR: ", pr[i,j])
                print("BOX1: ", pr[i, j, :4])
                print("BOX2: ", pr[i, j, 5:9])
                gt_box = get_box_coord(gt[i, j, :4]) + [x_, y_]
                pr_box1 = get_box_coord(pr[i, j, :4]) + [x_, y_]
                pr_box2 = get_box_coord(pr[i, j, 5:9]) + [x_, y_]

                iou_box_1 = IoU(gt_box, pr_box1)
                iou_box_2 = IoU(gt_box, pr_box2)
                assert iou_box_1 <= 1 and iou_box_2 <= 1 and iou_box_1 * iou_box_2 >= 0,("Error in IoU computation", iou_box_1, iou_box_2 )

                # Get Box with highest IoU with the GT:
                box_idx = 1 if iou_box_1 >= iou_box_2 else 2
                gt_ = gt[i, j]
                if box_idx == 1:
                    pr_ = np.concatenate((pr[i, j, :5], pr[i, j, 10:]), axis=0)
                else:
                    pr_ = np.concatenate((pr[i, j, 5:10], pr[i, j, 10:]), axis=0)

                # Compute Squared Error Loss for each component:
                # position
                loss += lamb_obj * (np.square(gt_[0] - pr_[0]) + np.square(gt_[1] - pr_[1]))
                # dimensions
                loss += lamb_obj * (np.square(np.sqrt(gt_[2]) - np.sqrt(pr_[2])) + np.square(np.sqrt(gt_[3]) - np.sqrt(pr_[3])))
                # P(Obj) cell with obj
                loss += np.square(1 - pr_[4])

                # P(C_i|Obj)
                try:
                    for idx in range(C):
                        if gt_[4 + idx] == 1:
                            loss += np.square(1 - pr_[5 + idx])
                        else:
                            loss += np.square(0 - pr_[5 + idx])
                except:
                    print(pr_)

            else:
                # P(Obj) cell without obj
                # not clear from the paper. Maybe change to the BBox with highest confidence, or both of them.
                loss += lamb_noobj * np.square(0 - 1)
