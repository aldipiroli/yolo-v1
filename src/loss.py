import torch
import numpy as np
import random
import math
from random import randrange
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from src.utils.utils import IoU, split_output_boxes


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
        box_present = gt[..., 4:5]

        pr_box0, pr_box1 = split_output_boxes(pr)

        iou_box0 = torch.unsqueeze(IoU(gt, pr_box0), dim=-1)
        iou_box1 = torch.unsqueeze(IoU(gt, pr_box1), dim=-1)

        ious = torch.cat((iou_box0, iou_box1), dim=3)
        _, argmax_iou = torch.max(ious, dim=3)
        argmax_iou = argmax_iou.unsqueeze(-1)

        pred = (1 - argmax_iou) * pr_box0 + argmax_iou * pr_box1
        loss = 0

        # ========================================= #
        # Loss position x,y:
        # ========================================= #  
        gt_ = gt[..., 0:2]
        pred_ = pred[..., 0:2]
        # loss += self.mse(gt_, pred_)

        # ========================================= #
        # Loss position h,w:
        # ========================================= #  
        gt_ = torch.sqrt(torch.abs(gt[..., 2:4]))
        pred_ = torch.sqrt(pred[..., 2:4])
        # loss += self.mse(gt_, pred_)


        # ========================================= #
        # Loss obj:
        # ========================================= #  
        gt_ = gt[..., 4:5]
        pred_ = pred[..., 4:5]
        loss += self.mse(gt_, pred_)


        # ========================================= #
        # Loss no obj:
        # ========================================= #  
        gt_ = gt[..., 4:5]
        pred_ = pred[..., 4:5]
        loss += self.mse(gt_, pred_)



        print("Loss: ", loss)
        print("gt: ", gt_[..., 0])
        print("pr: ", pred_[..., 0])



        print("-" * 30)
        print("box_present.shape: ", box_present.shape)
        print("pr_box1.shape: ", pr_box1.shape)
        print("iou_box1.shape: ", iou_box1.shape)
        print("ious.shape: ", ious.shape)
        print("argmax_iou.shape: ", argmax_iou.shape)
        print("pred.shape: ", pred.shape)
        print("gt.shape: ", gt.shape)
        print("pred.shape: ", pred.shape)
        print("gt_.shape: ", gt_.shape)
        print("pred_.shape: ", pred_.shape)
        return 0
