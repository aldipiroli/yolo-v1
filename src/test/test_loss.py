import unittest
import numpy as np
import random
from random import randrange

from src.loss import *
from src.utils.plot import *

class TestLoss(unittest.TestCase):

    def make_gt(self, S=7, C=20, W=448, H=448, single_box=False):
        gt = np.zeros((S, S, 5))

        if not single_box:
            for i in range(S):
                for j in range(S):
                    if i > 2 and i < 5 and j > 2 and j < 5:
                        x = random.uniform(0.0, 1.0)
                        y = random.uniform(0.0, 1.0)
                        w = random.uniform(0.0, 50)
                        h = random.uniform(0.0, 50)
                        rand_pr = random.uniform(0.0, 1.0)
                        if rand_pr > 0.5:
                            rnd_cls = randrange(C)
                        else:
                            rnd_cls = -1
                        gt[i, j] = [x, y, w, h, rnd_cls]
                    else:
                        gt[i, j] = [0, 0, 0, 0, -1]
        else:
            gt[:, :, 4] = -1
            gt[2, 2, :] = [0, 0, 50, 100, 10]

        return gt

    def make_pr(self, gt, B=2, err=5, single_box=False):
        S = gt.shape[0]
        pr = np.zeros((S, S, B * 5))
        pr[:, :, 4] = -1
        pr[:, :, 9] = -1

        if not single_box:
            for i in range(S):
                for j in range(S):
                    if gt[i, j, 4] != -1:
                        rand_pr = random.uniform(0.0, 1.0)
                        if rand_pr > 0.3:
                            pr[i, j, :5] = gt[i, j, :] + random.uniform(0.0, err)
                            pr[i, j, 4] = int(pr[i, j, 4])
                            pr[i, j, 5:] = gt[i, j, :] + random.uniform(0.0, err)
                            pr[i, j, 9] = int(pr[i, j, 9])
        else:
            pr[2, 2, :5] = [0, 0, 105, 40, 10]
            pr[2, 2, 5:] = [0, 0, 50, 99, 10]

        return pr

    def test_loss(self):
        single_box = False
        gt = self.make_gt(single_box=False)
        pr = self.make_pr(gt, single_box=False)

        fig, ax = plot_boxes(gt, color="red")
        fig, ax = plot_boxes(pr[:, :, :5], color="lime", fig=fig, ax=ax)
        fig, ax = plot_boxes(pr[:, :, 5:], color="orange", fig=fig, ax=ax)
        plt.show()
        


if __name__ == '__main__':
    unittest.main()
