import torch
from torch import nn
import numpy as np
import random


class YOLOv1(nn.Module):
    def __init__(self):
        super(YOLOv1, self).__init__()
        self.conv_layer_1 = nn.Sequential(nn.Conv2d(3, 64, [7, 7], 2, 3), nn.MaxPool2d(2, 2))
        self.conv_layer_2 = nn.Sequential(nn.Conv2d(64, 192, [3, 3], 1, 1), nn.MaxPool2d(2, 2))
        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(192, 128, [1, 1], 1, 0),
            nn.Conv2d(128, 256, [3, 3]),
            nn.Conv2d(256, 256, [1, 1], 1, 1),
            nn.Conv2d(256, 512, [3, 3], 1, 1),
            nn.MaxPool2d(2, 2),
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
        )

        self.conv_layer_6 = nn.Sequential(
            nn.Conv2d(1024, 1024, [3, 3], 1, 1),
            nn.Conv2d(1024, 1024, [3, 3], 1, 1),
        )

        self.fc_layer_1 = nn.Linear(50176, 4096)
        self.fc_layer_2 = nn.Linear(4096, 1470)

    def forward(self, x):
        y = self.conv_layer_1(x)
        y = self.conv_layer_2(y)
        y = self.conv_layer_3(y)
        y = self.conv_layer_4(y)
        y = self.conv_layer_5(y)
        y = self.conv_layer_6(y)
        y = self.fc_layer_1(torch.flatten(y))
        y = self.fc_layer_2(y)
        y = torch.reshape(y, (7, 7, 30))

        return y


if __name__ == "__main__":
    nn = YOLOv1()

    x = np.random.rand(1, 3, 448, 448)
    x = torch.Tensor(x)
    y = nn(x)
    print("Input shape: ", x.shape)
    print("Output shape: ", y.shape)
