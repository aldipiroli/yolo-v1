import unittest
import numpy as np
import random
import torch
from random import randrange
import matplotlib.pyplot as plt


from src.loss import compute_loss
from src.utils.plot import plot_boxes


class TestLoss(unittest.TestCase):
    def make_gt(self, batch_size=2, S=7, C=20, W=448, H=448, manual_input=False):
        gts = []

        for _ in range(batch_size):
            counter = 0
            gt = np.zeros((S, S, 24))

            for i in range(S):
                for j in range(S):

                    if i > 2 and i < 5 and j > 2 and j < 5:
                        x = random.uniform(0.0, 1.0)
                        y = random.uniform(0.0, 1.0)
                        w = random.uniform(0.0, 50)
                        h = random.uniform(0.0, 50)

                        # Create a box wit probability 0.5
                        if random.uniform(0.0, 1.0) > 0.5:
                            gt[i, j, :4] = [x, y, w, h]
                            rnd_cls = randrange(C)
                            gt[i, j, 4 + rnd_cls] = 1
                            counter += 1
            gts.append(gt)

        return np.array(gts)

    def make_pr(self, gts, S=7, B=2, C=20, err=15, manual_input=False):
        prs = []
        for k in range(gts.shape[0]):
            pr = np.zeros((S, S, B * 5 + C))
            gt = gts[k, :]

            for i in range(S):
                for j in range(S):
                    if gt[i, j, 4:].any() != 0:
                        # Box 1:
                        pr[i, j, :2] = np.clip(gt[i, j, :2] + random.uniform(-err, err), 0.1, 1)
                        pr[i, j, 2:4] = np.clip(gt[i, j, 2:4] + random.uniform(-err, err), 1, err)
                        pr[i, j, 4] = 0.33

                        # Box 2:
                        pr[i, j, 5:9] = gt[i, j, :4] + random.uniform(-err, err)
                        pr[i, j, 9] = 0.85

                        # Random class
                        pr[i, j, 10:] = 0
                        rnd_cls = randrange(C)
                        pr[i, j, 10 + rnd_cls] = 1
            prs.append(pr)

        return np.array(prs)

    def test_loss(self):
        manual_input = False
        gt = self.make_gt(manual_input=manual_input)
        pr = self.make_pr(gt, manual_input=manual_input)

        compute_loss(gt, pr)

        # # Plot GT
        # batch_idx = 0
        # fig, ax = plot_boxes(gt[batch_idx, :], color="red")

        # # Plot Box 1
        # pred_box = np.concatenate((pr[batch_idx, :, :, :4], pr[batch_idx, :, :, 10:]), axis=2)
        # fig, ax = plot_boxes(pred_box, color="lime", fig=fig, ax=ax)

        # # Plot Box 2
        # pred_box = np.concatenate((pr[batch_idx, :, :, 5:9], pr[batch_idx, :, :, 10:]), axis=2)
        # fig, ax = plot_boxes(pred_box, color="aqua", fig=fig, ax=ax)
        # plt.show()


if __name__ == "__main__":
    unittest.main()
