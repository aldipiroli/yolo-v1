import unittest
import numpy as np
import torch
from src.network import YOLOv1

class TestDataLoader(unittest.TestCase):
    S, B, C = 7, 2, 20
    nn = YOLOv1(S, B, C)

    # Generate Fake Data:
    N_BATCH = 5
    x = np.random.rand(N_BATCH, 3, 448, 448)
    x = torch.Tensor(x)

    # Network Output:
    y = nn(x)

    assert list(y.shape) == [N_BATCH, S, S, B*5 + C], ("Network Output Shape does not match expectation, ", list(y.shape))
    print("Network Output: ", list(y.shape))


if __name__ == "__main__":
    unittest.main()
