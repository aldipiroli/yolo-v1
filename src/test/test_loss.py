import unittest
import numpy as np
import random
from random import randrange
import matplotlib.pyplot as plt


from src.loss import compute_loss
from src.utils.plot import plot_boxes


class TestLoss(unittest.TestCase):
    def make_gt(self, S=7, C=20, W=448, H=448, manual_input=False):
        counter = 0
        gt = np.zeros((S, S, 24))
        
        if not manual_input:
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

        else:
            gt[2, 2, :4] = [0, 0, 50, 100]
            rnd_cls = 1
            gt[2, 2, 4 + rnd_cls] = 1
            
        return gt, counter

    def make_pr(self, gt, B=2, C=20, err=5, manual_input=False):
        S = gt.shape[0]
        pr = np.zeros((S, S, B * 5 + C))
        counter = 0

        if not manual_input:
            for i in range(S):
                for j in range(S):
                    if gt[i, j, 4:].any() != 0:
                        # Box 1:
                        pr[i, j, :4] = gt[i, j, :4] + random.uniform(-err, err)
                        pr[i, j, 4]  = 0.33

                        # Box 2:
                        pr[i, j, 5:9] = gt[i, j, :4] + random.uniform(-err, err)
                        pr[i, j, 9]  = 0.85

                        # Random class
                        pr[i, j, 10:] = 0
                        rnd_cls = randrange(C)
                        pr[i, j, 10 + rnd_cls] = 1

                        counter += 1

        else:
            pr[2, 2, :5] = [0, 0, 70, 20, 0.88]
            pr[2, 2, 5:10] = [0, 0, 20, 60, 0.97]

        return pr, counter

    def test_loss(self):
        manual_input = False
        gt, sum_gt = self.make_gt(manual_input=manual_input)
        pr, sum_pr = self.make_pr(gt, manual_input=manual_input)
        assert sum_gt == sum_pr,  "#GT Boxes != #PR Boxes"

        compute_loss(gt, pr)
        
        # # Plot GT
        # fig, ax = plot_boxes(gt, color="red")

        # # Plot Box 1
        # pred_box = np.concatenate((pr[:, :, :4], pr[:, :, 10:]), axis=2)
        # fig, ax = plot_boxes(pred_box, color="lime", fig=fig, ax=ax)

        # # Plot Box 2
        # pred_box = np.concatenate((pr[:, :, 5:9], pr[:, :, 10:]), axis=2)
        # fig, ax = plot_boxes(pred_box, color="aqua", fig=fig, ax=ax)
        # plt.show()


if __name__ == "__main__":
    unittest.main()
