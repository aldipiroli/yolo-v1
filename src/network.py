import torch
import random
from torch import nn
import numpy as np

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class YOLOv1(nn.Module):
    def __init__(self, split_size, blocks_num, num_classes):
        """
        split_size (S): size of the grid that the image is divided into
        block_nums (B): number of boxes predicted per cell by the network
        num_classes (C): number of classes that the network is trained 
        """
        super(YOLOv1, self).__init__()

        self.S, self.B, self.C = split_size, blocks_num, num_classes

        self.conv_layer_1 = nn.Sequential(nn.Conv2d(3, 64, [7, 7], 2, 3), nn.MaxPool2d(2, 2))
        self.conv_layer_2 = nn.Sequential(nn.Conv2d(64, 192, [3, 3], 1, 1), nn.MaxPool2d(2, 2))
        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(192, 128, [1, 1], 1, 0),
            nn.Conv2d(128, 256, [3, 3]),
            nn.Conv2d(256, 256, [1, 1], 1, 1),
            nn.Conv2d(256, 512, [3, 3], 1, 1),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.1),
        )

        self.conv_layer_4 = nn.Sequential(
            # repeat x4: #1
            nn.Conv2d(512, 256, [1, 1], 1, 0),
            nn.Conv2d(256, 512, [3, 3], 1, 1),
            # repeat x4: #2
            nn.Conv2d(512, 256, [1, 1], 1, 0),
            nn.Conv2d(256, 512, [3, 3], 1, 1),
            # repeat x4: #3
            nn.Conv2d(512, 256, [1, 1], 1, 0),
            nn.Conv2d(256, 512, [3, 3], 1, 1),
            # repeat x4: #4
            nn.Conv2d(512, 256, [1, 1], 1, 0),
            nn.Conv2d(256, 512, [3, 3], 1, 1),
            nn.Conv2d(512, 512, [1, 1], 1, 0),
            nn.Conv2d(512, 1024, [3, 3], 1, 1),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.1),
        )

        self.conv_layer_5 = nn.Sequential(
            # repeat x2: #1
            nn.Conv2d(1024, 512, [1, 1], 1, 0),
            nn.Conv2d(512, 1024, [3, 3], 1, 1),
            # repeat x2: #2
            nn.Conv2d(1024, 512, [1, 1], 1, 0),
            nn.Conv2d(512, 1024, [3, 3], 1, 1),
            nn.Conv2d(1024, 1024, [3, 3], 1, 1),
            nn.Conv2d(1024, 1024, [3, 3], 2, 1),
            nn.LeakyReLU(0.1),
        )

        self.conv_layer_6 = nn.Sequential(nn.Conv2d(1024, 1024, [3, 3], 1, 1), nn.Conv2d(1024, 1024, [3, 3], 1, 1), nn.LeakyReLU(0.1))

        self.fc_layer_1 = nn.Sequential(nn.Flatten(), nn.Linear(1024 * self.S * self.S, 4096))
        self.fc_layer_2 = nn.Sequential(nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C)), nn.ReLU())

    def forward(self, x):
        y = self.conv_layer_1(x)
        y = self.conv_layer_2(y)
        y = self.conv_layer_3(y)
        y = self.conv_layer_4(y)
        y = self.conv_layer_5(y)
        y = self.conv_layer_6(y)
        y = self.fc_layer_1(y)
        y = self.fc_layer_2(y)
        y = torch.reshape(y, (-1, self.S, self.S, 30))

        return y
    
