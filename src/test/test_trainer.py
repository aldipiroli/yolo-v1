import unittest
import yaml 
import torch

from src.trainer import Trainer
from src.loss import YOLOv1Loss

class TestTrainer(unittest.TestCase):
    def read_config(conf_file):
        conf = []
        with open(conf_file, "r") as stream:
            try:
                conf = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return conf
    
    # Set up network:
    conf = read_config(conf_file="config/yolo_v1.yaml")
    trainer = Trainer(conf)
    trainer.train()


if __name__ == "__main__":
    unittest.main()
