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


class Trainer:
    def __init__(self, conf):
        self.conf = conf
        self.N_EPOCH = conf["TRAINING"]["n_epochs"]
        self.ROOT_DIR = conf["DATASET"]["root_dir"]


        self.dataset = DatasetVOC2007(root_dir=self.ROOT_DIR, split="train")
        self.data_loader = DataLoader(
            self.dataset, batch_size=self.conf["TRAINING"]["batch_size"], 
            num_workers=self.conf["TRAINING"]["num_workers"],
            shuffle=True
        )
        self.S = self.conf["NETWORK"]["S"]
        self.B = self.conf["NETWORK"]["B"]
        self.C = self.conf["NETWORK"]["C"]

        self.net = YOLOv1(self.S, self.B, self.C)
        self.net.apply(weights_init)

        self.loss = YOLOv1Loss()

        self.learning_rate = self.conf["TRAINING"]["learning_rate"]
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
    


    def train_overfit(self):
        for epoch in range(self.N_EPOCH):

            img, label = self.dataset[3]
            img_ = img.transpose(2,0).transpose(0,1)

            self.optimizer.zero_grad()

            out = self.net(img.unsqueeze(0))
            print("The output: ", out)
            label = label.unsqueeze(0)
            loss = self.loss(label, out)
            print("Loss ->", loss)
            loss.backward()
            self.optimizer.step()

            # with torch.no_grad():
            #     box1, box2 = split_output_boxes(out)
            #     fig, ax = plot_voc2007_labels(img_,  label[0, ...])             
            #     plot_voc2007_labels(img_,  box1[0, :].detach().numpy(), fig=fig, ax=ax, color="green")             
            #     plt.show()
            input("....")


    def train(self):
        for epoch in range(self.N_EPOCH):
            # for i, (img, label) in enumerate(self.data_loader):
            img, label = self.dataset[3]
            img_ = img.transpose(2,0).transpose(0,1)
            
            out = self.net(img.unsqueeze(0))
            box1, box2 = split_output_boxes(out)
            print("Out: ", out.shape)
            print("Box1: ", box1.shape)
            print("Box2: ", box2.shape)
            print("label: ", label.shape)

            fig, ax = plot_voc2007_labels(img_,  label)             
            plot_voc2007_labels(img_,  box1[0, :].detach().numpy(), fig=fig, ax=ax, color="green")             
            plt.show()
            input("...")
