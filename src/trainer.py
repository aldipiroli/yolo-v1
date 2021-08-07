import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn

from src.network import YOLOv1, weights_init
from src.dataset import DatasetVOC2007
from src.loss import YOLOv1Loss
from src.utils.plot import plot_voc2007_labels
from src.utils.utils import split_output_boxes


def make_fake_output(label):
    box_present = label[..., 4:5]

    box_1 = box_present * (label[..., :4] + 2.5)

    box_2 = label[..., :4].clone()
    box_2[..., 0] = 30
    box_2[..., 1] = 50
    box_2[..., 2] = 50
    box_2[..., 3] = 180

    box_2 = box_present * box_2

    out = torch.zeros((1, 7, 7, 30))
    out[..., :4] = box_2
    out[..., 4:5] = box_present * 0.8


    out[..., 5:9] = box_1
    out[..., 9:10] = box_present * 0.3

    return out


class Trainer:
    def __init__(self, conf):
        self.conf = conf
        self.N_EPOCH = conf["TRAINING"]["n_epochs"]
        self.ROOT_DIR = conf["DATASET"]["root_dir"]
        self.BATCH_SIZE = self.conf["TRAINING"]["batch_size"]
        self.NUM_WORKERS = self.conf["TRAINING"]["num_workers"]

        self.S = self.conf["NETWORK"]["S"]
        self.B = self.conf["NETWORK"]["B"]
        self.C = self.conf["NETWORK"]["C"]

        self.dataset = DatasetVOC2007(root_dir=self.ROOT_DIR, split="train")
        self.data_loader = DataLoader(
            self.dataset, batch_size=self.BATCH_SIZE, num_workers=self.NUM_WORKERS, shuffle=True
        )

        self.net = YOLOv1(self.S, self.B, self.C)
        self.net.apply(weights_init)
        self.loss = YOLOv1Loss()

        self.learning_rate = self.conf["TRAINING"]["learning_rate"]
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def train(self):
        self.net.zero_grad()
        for epoch in range(self.N_EPOCH):
            print("\n============= Epoch: %d =============\n" % epoch)
            img, label = self.dataset[3]

            # Make a batch:
            img = img.unsqueeze(0)
            label = label.unsqueeze(0)

            # Make a fake output:
            # out = make_fake_output(label)
            out = self.net(img)
            # box1, box2 = split_output_boxes(out)

            # Compute Loss:
            loss = self.loss(label, out)

            # Backprop:
            # self.net.zero_grad()
            self.optimizer.zero_grad()   # zero the gradient buffers
            loss.backward()
            self.optimizer.step()

            # Plot img and label:
            # img_ = img[0].transpose(2, 0).transpose(0, 1)
            # fig, ax = plot_voc2007_labels(img_, label[0])
            # fig, ax = plot_voc2007_labels(img_, box1[0], fig=fig, ax=ax, color="lime")
            # plot_voc2007_labels(img_, box2[0], fig=fig, ax=ax, color="blue")
            # plt.savefig("debug_image.png")

            # Debug printing:
            print("-" * 30)
            print("img.shape", img.shape)
            print("label.shape", label.shape)
            print("out.shape", out.shape)
            print("loss_val:", loss)
            input("....")
