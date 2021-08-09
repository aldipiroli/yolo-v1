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


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class YOLOv1(nn.Module):
    def __init__(self, split_size, blocks_num, num_classes):
        """
        split_size (S): size of the grid that the image is divided into
        block_nums (B): number of boxes predicted per cell by the network
        num_classes (C): number of classes that the network is trained 
        """
        super(YOLOv1, self).__init__()

        self.S, self.B, self.C = split_size, blocks_num, num_classes
        self.layers = []

        # Layer 1:
        # self.conv_layer_1 = nn.Sequential(nn.Conv2d(3, 64, [7, 7], 2, 3), nn.MaxPool2d(2, 2))
        self.layers += [CNNBlock(3, 64, kernel_size = 7, stride=2, padding=3)]
        self.layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        # Layer 2:
        # self.conv_layer_2 = nn.Sequential(nn.Conv2d(64, 192, [3, 3], 1, 1), nn.MaxPool2d(2, 2))
        self.layers += [CNNBlock(64, 192, kernel_size = 3, stride=1, padding=1)]
        self.layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        # Layer 3:
        # self.conv_layer_3 = nn.Sequential(
        #     nn.Conv2d(192, 128, [1, 1], 1, 0),
        #     nn.Conv2d(128, 256, [3, 3]),
        #     nn.Conv2d(256, 256, [1, 1], 1, 1),
        #     nn.Conv2d(256, 512, [3, 3], 1, 1),
        #     nn.MaxPool2d(2, 2),
        #     nn.LeakyReLU(0.1),
        # )
        self.layers += [CNNBlock(192, 128, kernel_size = 1, stride=1, padding=0)]
        self.layers += [CNNBlock(128, 256, kernel_size = 3)]
        self.layers += [CNNBlock(256, 256, kernel_size = 1, stride=1, padding=1)]
        self.layers += [CNNBlock(256, 512, kernel_size = 3, stride=1, padding=1)]
        self.layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        # Layer 4:
        # self.conv_layer_4 = nn.Sequential(
        #     # repeat x4: #1
        #     nn.Conv2d(512, 256, [1, 1], 1, 0),
        #     nn.Conv2d(256, 512, [3, 3], 1, 1),
        #     # repeat x4: #2
        #     nn.Conv2d(512, 256, [1, 1], 1, 0),
        #     nn.Conv2d(256, 512, [3, 3], 1, 1),
        #     # repeat x4: #3
        #     nn.Conv2d(512, 256, [1, 1], 1, 0),
        #     nn.Conv2d(256, 512, [3, 3], 1, 1),
        #     # repeat x4: #4
        #     nn.Conv2d(512, 256, [1, 1], 1, 0),
        #     nn.Conv2d(256, 512, [3, 3], 1, 1),
        #     nn.Conv2d(512, 512, [1, 1], 1, 0),
        #     nn.Conv2d(512, 1024, [3, 3], 1, 1),
        #     nn.MaxPool2d(2, 2),
        #     nn.LeakyReLU(0.1),
        # )

        self.layers += [CNNBlock(512, 256, kernel_size = 1, stride=1, padding=0)]
        self.layers += [CNNBlock(256, 512, kernel_size = 3, stride=1, padding=1)]

        self.layers += [CNNBlock(512, 256, kernel_size = 1, stride=1, padding=0)]
        self.layers += [CNNBlock(256, 512, kernel_size = 3, stride=1, padding=1)]

        self.layers += [CNNBlock(512, 256, kernel_size = 1, stride=1, padding=0)]
        self.layers += [CNNBlock(256, 512, kernel_size = 3, stride=1, padding=1)]

        self.layers += [CNNBlock(512, 256, kernel_size = 1, stride=1, padding=0)]
        self.layers += [CNNBlock(256, 512, kernel_size = 3, stride=1, padding=1)]
        self.layers += [CNNBlock(512, 512, kernel_size = 1, stride=1, padding=0)]
        self.layers += [CNNBlock(512, 1024, kernel_size = 3, stride=1, padding=1)]
        self.layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]


        # Layer 5:
        # self.conv_layer_5 = nn.Sequential(
        #     # repeat x2: #1
        #     nn.Conv2d(1024, 512, [1, 1], 1, 0),
        #     nn.Conv2d(512, 1024, [3, 3], 1, 1),
        #     # repeat x2: #2
        #     nn.Conv2d(1024, 512, [1, 1], 1, 0),
        #     nn.Conv2d(512, 1024, [3, 3], 1, 1),
        #     nn.Conv2d(1024, 1024, [3, 3], 1, 1),
        #     nn.Conv2d(1024, 1024, [3, 3], 2, 1),
        #     nn.LeakyReLU(0.1),
        # )
        self.layers += [CNNBlock(1024, 512, kernel_size = 1, stride=1, padding=0)]
        self.layers += [CNNBlock(512, 1024, kernel_size = 3, stride=1, padding=1)]

        self.layers += [CNNBlock(1024, 512, kernel_size = 1, stride=1, padding=0)]
        self.layers += [CNNBlock(512, 1024, kernel_size = 3, stride=1, padding=1)]
        self.layers += [CNNBlock(1024, 1024, kernel_size = 3, stride=1, padding=1)]
        self.layers += [CNNBlock(1024, 1024, kernel_size = 3, stride=2, padding=1)]
        
        # Layer 6:
        # self.conv_layer_6 = nn.Sequential(nn.Conv2d(1024, 1024, [3, 3], 1, 1), nn.Conv2d(1024, 1024, [3, 3], 1, 1), nn.LeakyReLU(0.1))
        self.layers += [CNNBlock(1024, 1024, kernel_size = 3, stride=1, padding=1)]
        self.layers += [CNNBlock(1024, 1024, kernel_size = 3, stride=1, padding=1)]

        # Fully Conn:
        # self.fc_layer_1 = nn.Sequential(nn.Flatten(), nn.Linear(1024 * self.S * self.S, 4096))
        # self.fc_layer_2 = nn.Sequential(nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C)), nn.ReLU())
        self.layers += [nn.Flatten(), nn.Linear(1024 * self.S * self.S, 4096)]
        self.layers += [nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C)), nn.ReLU()]
        self.net = nn.Sequential(*self.layers)


    def forward(self, x):
        # y = self.conv_layer_1(x)
        # y = self.conv_layer_2(y)
        # y = self.conv_layer_3(y)
        # y = self.conv_layer_4(y)
        # y = self.conv_layer_5(y)
        # y = self.conv_layer_6(y)
        # y = self.fc_layer_1(y)
        # y = self.fc_layer_2(y)
        y = self.net(x)
        y = torch.reshape(y, (-1, self.S, self.S, 30))

        return y
    
    
