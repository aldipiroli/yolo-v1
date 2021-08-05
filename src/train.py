import numpy as np
import torch
import yaml
from src.network import YOLOv1
from src.dataset import DatasetVOC2007
from torch.utils.data import DataLoader


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

    def train(self):
        for epoch in range(self.N_EPOCH):
            for i, (img, label) in enumerate(self.data_loader):
                print(i, img.shape, label.shape)


if __name__ == "__main__":
    print(" *** Started Training *** ")
    CONFIG_FILE = "config/yolo_v1.yaml"

    conf = []
    with open(CONFIG_FILE, "r") as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    trainer = Trainer(conf)
    trainer.train()
