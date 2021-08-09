import torch
import numpy as np

def get_box_coord(box):
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
    box_x, box_y = box[..., 0], box[..., 1]
    box_w, box_h = box[..., 2], box[..., 3]

    box_x_min = box_x - box_w / 2
    box_y_min = box_y - box_h / 2

    box_x_max = box_x + box_w / 2
    box_y_max = box_y + box_h / 2

    return [box_x_min, box_y_min, box_x_max, box_y_max]

def IoU(gt, pr):
    """
    Intersection Box:
    A------------------.
    |                  |
    |                  |
    |                  |
    .------------------B
    """

    gt_boxs = get_box_coord(gt)
    pr_boxs = get_box_coord(pr)

    # Find the corner:
    A_x = torch.max(gt_boxs[0], pr_boxs[0])
    A_y = torch.max(gt_boxs[1], pr_boxs[1])

    B_x = torch.min(gt_boxs[2], pr_boxs[2])
    B_y = torch.min(gt_boxs[3], pr_boxs[3])

    # Compute Intersection:
    inter = (B_x - A_x).clamp(0) * (B_y - A_y).clamp(0)

    # Compute union:
    gt_w = gt[..., 2]
    gt_h = gt[..., 3]

    pr_w = pr[..., 2]
    pr_h = pr[..., 3]

    # Compute Union:
    gt_area = abs(gt_w * gt_h)
    pr_area = abs(pr_w * pr_h)
    union = gt_area + pr_area - inter

    # Compute IoU:
    IoU = inter / union

    # Remove wrong IoUs:
    IoU[torch.isnan(IoU)] = 0

    mask_neg = torch.le(IoU, 0 - 1e-6)
    mask_one = torch.gt(IoU, 1)

    # Check if something is wrong with the IoU computation:
    if mask_neg.any() == True or mask_neg.any() == True or mask_one.any() == True or mask_one.any() == True:
        print("Error in the IoU computation")
        print("\n\nIoU: ", IoU, " \n\ninter: ",inter,"\n\nunion", union )
        exit()
    return IoU

def split_output_boxes(output):
    box1 = torch.cat((output[..., :5], output[..., 10:]), 3)
    box2 = torch.cat((output[..., 5:10], output[..., 10:]), 3)

    return box1, box2