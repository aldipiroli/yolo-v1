import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim

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

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("TRAINING ON DEVICE: ", self.device)

        self.dataset = DatasetVOC2007(root_dir=self.ROOT_DIR, split="train")
        self.data_loader = DataLoader(
            self.dataset, batch_size=self.BATCH_SIZE, num_workers=self.NUM_WORKERS, shuffle=True
        )

        self.net = YOLOv1(self.S, self.B, self.C).to(self.device)
        self.net.apply(weights_init)
        self.loss = YOLOv1Loss()

        self.learning_rate = self.conf["TRAINING"]["learning_rate"]
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=10e-5, weight_decay=0.0001)

    def train(self):
        torch.autograd.set_detect_anomaly(True)
        self.net.zero_grad()
        self.net.train()

        for epoch in range(self.N_EPOCH):
            print("\n============= Epoch: %d =============\n" % epoch)
            for i, (img, label) in enumerate(self.data_loader):

                # Make a batch:
                img = img.to(self.device)
                label = label.to(self.device)

                print("img: ", img.shape)
                print("label: ", label.shape)

                out = self.net(img)

                # Compute Loss and Optimize:
                self.optimizer.zero_grad()  # zero the gradient buffers
                loss = self.loss(label, out, self.device)
                loss.backward()
                self.optimizer.step()

                if i % 100 == 0:
                    print("i: ", i, ", Loss: ", loss)
                    # Plot a sample:
                    with torch.no_grad():
                        img_ = img[0].cpu().transpose(2, 0).transpose(0, 1)
                        fig, ax = plot_voc2007_labels(img_, label[0].cpu())

                        box0, box1 = split_output_boxes(out.detach())
                        fig, ax = plot_voc2007_labels(img_, box0[0], fig=fig, ax=ax, color="lime")
                        plot_voc2007_labels(img_, box1[0], fig=fig, ax=ax, color="blue")
                        plt.xlim([0, 500])
                        plt.ylim([0, 500])
                        plt.savefig("../img/epoch_" + str(epoch) + "_i" + str(i) + ".png")
                        plt.close("all")

            # Save the model:
            MODEL_PATH = "../model/"
            torch.save(self.net.state_dict(), MODEL_PATH)
