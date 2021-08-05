import torch
import numpy as np
import random
import math
from random import randrange
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

from src.utils.utils import IoU


class YOLOv1Loss(torch.nn.Module):
    def __init__(self, C=20, B=2, S=7):
        super(YOLOv1Loss, self).__init__()
        self.C = C
        self.B = B
        self.S = S
        self.lamb_obj = 5
        self.lamb_noobj = 0.5
        self.mse = torch.nn.MSELoss(reduction="sum")
        self.loss = 0

    def forward(self, gt, pr):
        self.loss = 0

        exist_object = gt[..., 4].unsqueeze(-1)
        pr_box1 = pr[..., 0:5]
        pr_box2 = pr[..., 5:10]

        iou_box1 = torch.unsqueeze(IoU(gt, pr_box1), dim=-1)
        iou_box2 = torch.unsqueeze(IoU(gt, pr_box2), dim=-1)

        ious = torch.cat((iou_box1, iou_box2), dim=-1)
        max_iou, best_box = torch.max(ious, dim=-1)

        best_box = best_box.unsqueeze(-1)

        print("exist_object", exist_object.shape)
        print("pr", pr.shape)
        print("gt", gt.shape)
        print("best_box", best_box.shape)
        print("pr_box1", pr_box1.shape)
        print("pr_box2", pr_box2.shape)

        # Find labels and preds:
        pred_ = exist_object * ((1 - best_box) * pr_box1 + best_box * pr_box2)
        gt_ = exist_object * gt

        # ------------------------------- #
        # Loss x,y:
        # ------------------------------- #
        pred = pred_[..., 0:2]
        gt = gt_[..., 0:2]
        self.loss += self.lamb_obj * self.mse(pred, gt)

        # ------------------------------- #
        # Loss w,h:
        # ------------------------------- #
        pred = torch.sign(pred_[..., 2:4]) * torch.sqrt(torch.abs(pred_[..., 2:4]))
        gt = torch.sqrt(gt_[..., 2:4])
        self.loss += self.mse(pred, gt)

        # ------------------------------- #
        # Loss obj:
        # ------------------------------- #
        pred = pred_[..., 4:5]
        gt = exist_object * gt_[..., 4:5]
        self.loss += self.mse(pred, gt)

        return self.loss

        # # pred_x = Iobj* ((1-best_box) * pr_box1[...,0] + best_box*pr_box2[...,0])
        # # pred_y = Iobj* ((1-best_box) * pr_box1[...,1] + best_box*pr_box2[...,1])
        # # gt_x = Iobj * gt[..., 0]
        # # gt_y = Iobj * gt[..., 1]

        # print("pred", pred.shape)
        # print(pred.shape, pr_box1.shape)
        # print(loss.shape)

    # def forward(self, pr_boxes, gt_boxes):
    #     assert gt_boxes.shape[0] == pr_boxes.shape[0], (
    #         "GT and PR have different batch sizes!",
    #         gt_boxes.shape[0],
    #         pr_boxes.shape[0],
    #     )
    #     N_BATCHES = gt_boxes.shape[0]

    #     LAMB_OBJ = 5
    #     LAMB_NOOBJ = 0.5

    #     losses = []
    #     for n in range(N_BATCHES):
    #         gt = gt_boxes[n, :]
    #         pr = pr_boxes[n, :]

    #         S = gt.shape[0]
    #         loss = 0
    #         for i in range(S):
    #             for j in range(S):
    #                 # Check if an object is present in the cell:
    #                 print("The ground truth -->", gt[i, j])
    #                 if gt[i, j, 4:].any() != 0:
    #                     gt_box = get_box_coord(gt[i, j, :4])
    #                     pr_box1 = get_box_coord(pr[i, j, :4])
    #                     pr_box2 = get_box_coord(pr[i, j, 5:9])

    #                     iou_box1 = IoU(gt_box, pr_box1)
    #                     iou_box2 = IoU(gt_box, pr_box2)

    #                     # Select box with highest IoU with gt:
    #                     if iou_box1 >= iou_box2:
    #                         pred = torch.cat((pr[i, j, :5], pr[i, j, 10:]), axis=0)
    #                     else:
    #                         pred = torch.cat((pr[i, j, 5:10], pr[i, j, 10:]), axis=0)
    #                     gt_ = gt[i, j]

    #                     # Compute Squared Error Loss for each component:
    #                     # position
    #                     loss += LAMB_OBJ * (torch.square(gt_[0] - pred[0]) + torch.square(gt_[1] - pred[1]))

    #                     # dimensions
    #                     loss += LAMB_OBJ * (
    #                         torch.square(torch.sqrt(gt_[2]) - torch.sqrt(pred[2]))
    #                         + torch.square(torch.sqrt(gt_[3]) - torch.sqrt(pred[3]))
    #                     )

    #                     # P(Obj) cell with obj
    #                     loss += torch.square(1 - pred[4])

    #                     # P(C_i|Obj)
    #                     for idx in range(self.C):
    #                         if gt_[4 + idx] == 1:
    #                             loss += torch.square(1 - pred[5 + idx])
    #                         else:
    #                             loss += torch.square(0 - pred[5 + idx])

    #                 else:
    #                     # P(Obj) cell without obj
    #                     # not clear from the paper.
    #                     loss += LAMB_NOOBJ * torch.square(0 - pr[i, j, 4])
    #                     loss += LAMB_NOOBJ * torch.square(0 - pr[i, j, 9])

    #         # Append batch loss
    #         loss = torch.tensor(loss)
    #         losses.append(loss)

    #     losses = torch.tensor(losses)
    #     loss = torch.mean(losses)
    #     return torch.tensor(loss, requires_grad=True)


# def compute_loss(gt_boxes, pr_boxes, C=20):
