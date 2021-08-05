import torch
import numpy as np


def get_box_coord(gt, pr):
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

    # Find Corners:
    # GT:
    gt_x, gt_y = gt[..., 0], gt[..., 1]
    gt_w, gt_h = gt[..., 2], gt[..., 3]

    gt_x_min = gt_x - gt_w / 2
    gt_y_min = gt_y - gt_h / 2

    gt_x_max = gt_x + gt_w / 2
    gt_y_max = gt_y + gt_h / 2

    # PR:
    pr_x, pr_y = pr[..., 0], pr[..., 1]
    pr_w, pr_h = pr[..., 2], pr[..., 3]

    pr_x_min = pr_x - pr_w / 2
    pr_y_min = pr_y - pr_h / 2

    pr_x_max = pr_x + pr_w / 2
    pr_y_max = pr_y + pr_h / 2

    # Find the corner:
    A_x = torch.max(gt_x_min, pr_x_min)
    A_y = torch.max(gt_y_min, pr_y_min)

    B_x = torch.min(gt_x_max, pr_x_max)
    B_y = torch.min(gt_y_max, pr_y_max)

    return A_x, A_y, B_x, B_y


def IoU(gt, pr):
    """
    Intersection Box:
    A------------------.
    |                  |
    |                  |
    |                  |
    .------------------B
    """

    A_x, A_y, B_x, B_y = get_box_coord(gt, pr)

    # Compute Intersection: 
    inter = (B_x - A_x) * (B_y - A_y)
    
    # Compute union:
    gt_w = gt[..., 2]
    gt_h = gt[..., 3]

    pr_w = pr[..., 2]
    pr_h = pr[..., 3]
    
    # Compute Union:
    gt_area = abs(gt_w * gt_h)
    pr_area = abs(pr_w * pr_h)
    union = gt_area + pr_area - inter

    # Compute IoU:
    IoU = inter / union

    # Remove wrong IoUs:
    IoU[torch.isnan(IoU)] = 0

    mask_neg = torch.le(IoU, 0 - 1e-6)
    mask_one = torch.gt(IoU, 1)

    if mask_neg.any() == True or mask_neg.any() == True:
        if mask_one.any() == True or mask_one.any() == True:
            print ("Error in the IoU computation")
    return IoU
