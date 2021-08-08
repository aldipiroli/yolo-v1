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
        self.mse = torch.nn.MSELoss()
        self.loss = 0

    def forward(self, gt, pr, device):
        box_present = gt[..., 4:5]

        pr_box0, pr_box1 = split_output_boxes(pr)

        iou_box0 = torch.unsqueeze(IoU(gt, pr_box0), dim=-1)
        iou_box1 = torch.unsqueeze(IoU(gt, pr_box1), dim=-1)

        ious = torch.cat((iou_box0, iou_box1), dim=3)
        _, argmax_iou = torch.max(ious, dim=3)
        argmax_iou = argmax_iou.unsqueeze(-1)

        pred = box_present * ((1 - argmax_iou) * pr_box0 + argmax_iou * pr_box1)
        loss = 0

        # ========================================= #
        # Loss position x,y:
        # ========================================= #
        gt_ = gt[..., 0:2]
        pred_ = pred[..., 0:2]
        loss += self.lamb_obj * torch.sum((gt_ - pred_)**2)

        # ========================================= #
        # Loss position h,w:
        # ========================================= #
        gt_ = torch.sqrt(torch.abs(gt[..., 2:4] +  1e-8))
        pred_ = torch.sqrt(torch.abs(pred[..., 2:4]+  1e-8))
        loss += self.lamb_obj * torch.sum((gt_ - pred_)**2)

        # ========================================= #
        # Loss obj:
        # ========================================= #
        gt_ = gt[..., 4:5]
        pred_ = pred[..., 4:5]
        loss += self.mse(gt_, pred_)

        # ========================================= #
        # Loss no obj:
        # ========================================= #
        box_no_present = 1 - gt[..., 4:5]
        pred_noobj_ = box_no_present * pr_box0[..., 4:5]
        gt_ = torch.zeros((pred_noobj_.shape))
        loss += self.lamb_noobj * torch.sum((gt_.to(device) - pred_noobj_)**2)

        # ========================================= #
        # Loss classes:
        # ========================================= #
        gt_ = gt[..., 5:]
        pred_ = pred[..., 5:]
        loss += torch.sum((gt_ - pred_)**2)

        return loss
