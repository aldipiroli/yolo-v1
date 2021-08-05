import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.network import YOLOv1
from src.dataset import DatasetVOC2007
from src.loss import YOLOv1Loss


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

        self.loss = YOLOv1Loss()

        self.learning_rate = self.conf["TRAINING"]["learning_rate"]
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)


    def train(self):
        for epoch in range(self.N_EPOCH):

            # for i, (img, label) in enumerate(self.data_loader):
            img, label = self.dataset[3]
            img = torch.unsqueeze(img, 0)
            label = torch.unsqueeze(label, 0)

            out = self.net(img)
            loss = compute_loss(label, out)
            print(img.shape, label.shape, out.shape, loss)
