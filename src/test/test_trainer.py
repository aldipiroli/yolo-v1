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

    # Overfit network in one single example:
    for i in range(1000):
        # Get input ready:
        img, label = trainer.dataset[3]
        img = torch.unsqueeze(img, 0)
        label = torch.unsqueeze(label, 0)

        out = trainer.net(img)
        print("Out: ", out.shape)
        print("label: ", label.shape)
        loss_fc = YOLOv1Loss()
        loss = loss_fc(label, out)
        print("The loss func: ", loss)
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()


        print("Step %d, loss %.2f" % (i, loss))



if __name__ == "__main__":
    unittest.main()
