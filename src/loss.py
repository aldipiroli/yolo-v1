import torch
import numpy as np
import random
import math
from random import randrange
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

from src.utils.utils import get_box_coord


def IoU(gt, pr):
    """
    Intersection Box:
    A------------------.
    |                  |
    |                  |
    |                  |
    .------------------B
    """
    # Compute intersection:
    A_x = max(gt[0, 0], pr[0, 0])
    A_y = max(gt[0, 1], pr[0, 1])

    B_x = min(gt[1, 0], pr[1, 0])
    B_y = min(gt[1, 1], pr[1, 1])

    inter = (B_x - A_x) * (B_y - A_y)
    if inter < 0:
        return 0

    # Compute union:
    gt_area = abs((gt[1, 0] - gt[0, 0]) * (gt[1, 1] - gt[0, 1]))
    pr_area = abs((pr[1, 0] - pr[0, 0]) * (pr[1, 1] - pr[0, 1]))
    union = gt_area + pr_area - inter

    IoU = inter / union

    assert IoU >= 0 and IoU <= 1, ("Error in the IoU computation, IoU: ", IoU)
    return IoU


def compute_loss(gt_boxes, pr_boxes, C=20):
    assert gt_boxes.shape[0] == pr_boxes.shape[0], ("GT and PR have different batch sizes!", gt_boxes.shape[0], pr_boxes.shape[0])
    N_BATCHES = gt_boxes.shape[0]

    LAMB_OBJ = 5
    LAMB_NOOBJ = 0.5

    losses = []
    for n in range(N_BATCHES):
        gt = gt_boxes[n, :]
        pr = pr_boxes[n, :]

        S = gt.shape[0]
        loss = 0
        for i in range(S):
            for j in range(S):
                # Check if an object is present in the cell:
                if gt[i, j, 4:].any() != 0:
                    gt_box = get_box_coord(gt[i, j, :4])
                    pr_box1 = get_box_coord(pr[i, j, :4])
                    pr_box2 = get_box_coord(pr[i, j, 5:9])

                    iou_box1 = IoU(gt_box, pr_box1)
                    iou_box2 = IoU(gt_box, pr_box2)

                    # Select box with highest IoU with gt:
                    if iou_box1 >= iou_box2:
                        pred = np.concatenate((pr[i, j, :5], pr[i, j, 10:]), axis=0)
                    else:
                        pred = np.concatenate((pr[i, j, 5:10], pr[i, j, 10:]), axis=0)
                    gt_ = gt[i, j]

                    # Compute Squared Error Loss for each component:
                    # position
                    loss += LAMB_OBJ * (np.square(gt_[0] - pred[0]) + np.square(gt_[1] - pred[1]))

                    # dimensions
                    loss += LAMB_OBJ * (np.square(np.sqrt(gt_[2]) - np.sqrt(pred[2])) + np.square(np.sqrt(gt_[3]) - np.sqrt(pred[3])))

                    # P(Obj) cell with obj
                    loss += np.square(1 - pred[4])

                    # P(C_i|Obj)
                    for idx in range(C):
                        if gt_[4 + idx] == 1:
                            loss += np.square(1 - pred[5 + idx])
                        else:
                            loss += np.square(0 - pred[5 + idx])

                else:
                    # P(Obj) cell without obj
                    # not clear from the paper.
                    loss += LAMB_NOOBJ * np.square(0 - pr[i, j, 4])
                    loss += LAMB_NOOBJ * np.square(0 - pr[i, j, 9])

        # Append batch loss
        losses.append(loss)

    return losses
