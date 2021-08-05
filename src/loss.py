import torch
import numpy as np
import random
import math
from random import randrange
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

class YOLOv1Loss(torch.nn.Module):
    def __init__(self, C=20):
        super(YOLOv1Loss,self).__init__()
        self.C = C

    def get_box_coord(self, box):
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
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        A = [x - w / 2, y - h / 2]
        B = [x + w / 2, y + h / 2]

        return torch.tensor((A, B))


    def IoU(self, gt, pr):
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

        # Sanity Check for the predictions:
        if pr_area == 0:
            return 0
        if pr[0, 0] < 0 or pr[0, 1] < 0 or pr[1, 0] < 0 or pr[1,1] < 0:
            return 0


        assert IoU >= 0 and IoU <= 1, ("Error in the IoU computation, IoU: ", IoU, gt, pr)
        return IoU

    def forward(self, pr_boxes, gt_boxes):
        assert gt_boxes.shape[0] == pr_boxes.shape[0], (
            "GT and PR have different batch sizes!",
            gt_boxes.shape[0],
            pr_boxes.shape[0],
        )
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
                    print("The ground truth -->", gt[i, j])
                    if gt[i, j, 4:].any() != 0:
                        gt_box = get_box_coord(gt[i, j, :4])
                        pr_box1 = get_box_coord(pr[i, j, :4])
                        pr_box2 = get_box_coord(pr[i, j, 5:9])

                        iou_box1 = IoU(gt_box, pr_box1)
                        iou_box2 = IoU(gt_box, pr_box2)

                        # Select box with highest IoU with gt:
                        if iou_box1 >= iou_box2:
                            pred = torch.cat((pr[i, j, :5], pr[i, j, 10:]), axis=0)
                        else:
                            pred = torch.cat((pr[i, j, 5:10], pr[i, j, 10:]), axis=0)
                        gt_ = gt[i, j]

                        # Compute Squared Error Loss for each component:
                        # position
                        loss += LAMB_OBJ * (torch.square(gt_[0] - pred[0]) + torch.square(gt_[1] - pred[1]))

                        # dimensions
                        loss += LAMB_OBJ * (
                            torch.square(torch.sqrt(gt_[2]) - torch.sqrt(pred[2]))
                            + torch.square(torch.sqrt(gt_[3]) - torch.sqrt(pred[3]))
                        )

                        # P(Obj) cell with obj
                        loss += torch.square(1 - pred[4])

                        # P(C_i|Obj)
                        for idx in range(self.C):
                            if gt_[4 + idx] == 1:
                                loss += torch.square(1 - pred[5 + idx])
                            else:
                                loss += torch.square(0 - pred[5 + idx])

                    else:
                        # P(Obj) cell without obj
                        # not clear from the paper.
                        loss += LAMB_NOOBJ * torch.square(0 - pr[i, j, 4])
                        loss += LAMB_NOOBJ * torch.square(0 - pr[i, j, 9])

            # Append batch loss
            loss = torch.tensor(loss)
            losses.append(loss)
        
        losses = torch.tensor(losses)
        loss = torch.mean(losses)
        return torch.tensor(loss, requires_grad=True)






# def compute_loss(gt_boxes, pr_boxes, C=20):