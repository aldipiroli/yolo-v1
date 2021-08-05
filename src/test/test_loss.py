import unittest
import torch 
import random 
import numpy as np 
import matplotlib.pyplot as plt
from src.utils.plot import plot_voc2007_labels


class TestYOLOv1Loss(unittest.TestCase):
    def make_gt(self, batch_size=2, S=7, C=20, W=448, H=448, manual_input=False):
        gts = []

        for _ in range(batch_size):
            counter = 0
            gt = np.zeros((S, S, 5 + C))

            for i in range(S):
                for j in range(S):

                    if i > 2 and i < 5 and j > 2 and j < 5:
                        x = random.uniform(0.0, 1.0)
                        y = random.uniform(0.0, 1.0)
                        w = random.uniform(0.0, 50)
                        h = random.uniform(0.0, 50)

                        # Create a box wit probability 0.5
                        if random.uniform(0.0, 1.0) > 0.5:
                            gt[i, j, :5] = [x, y, w, h, 1]
                            rnd_cls = random.randrange(C)
                            gt[i, j, 5 + rnd_cls] = 1
                            counter += 1
            gts.append(gt)
        gts = np.array(gts)
        gts = torch.from_numpy(gts)
        return gts
    
    def make_pr(self, gts, S=7, B=2, C=20, err=15, manual_input=False):
        prs = []
        for k in range(gts.shape[0]):
            pr = np.zeros((S, S, B * 5 + C))
            gt = gts[k, :]

            for i in range(S):
                for j in range(S):
                    if gt[i, j, 4] != 0:
                        # Box 1:
                        pr[i, j, :2] = np.clip(gt[i, j, :2] + random.uniform(-err, err), 0.1, 1)
                        pr[i, j, 2:4] = np.clip(gt[i, j, 2:4] + random.uniform(-err, err), 1, err)
                        pr[i, j, 4] = 0.33

                        # Box 2:
                        pr[i, j, 5:9] = gt[i, j, :4] + random.uniform(-err, err)
                        pr[i, j, 9] = 0.85

                        # Random class
                        pr[i, j, 10:] = 0
                        rnd_cls = random.randrange(C)
                        pr[i, j, 10 + rnd_cls] = 1
            prs.append(pr)
        prs = np.array(prs)
        prs = torch.from_numpy(prs)
        return np.array(prs)


    def test_loss(self):
        gt = self.make_gt()
        pr = self.make_pr(gt)

        img = np.random.random((448, 488))
        fig, ax = plot_voc2007_labels(img, gt[0, :], color="lime")
        fig, ax = plot_voc2007_labels(img, pr[0, :], fig=fig, ax=ax)
        plt.show()

if __name__ == "__main__":
    unittest.main()
