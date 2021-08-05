import unittest
from src.dataloader import DataLoaderVOC2007
from src.utils.plot import plot_voc2007_boxes, plot_voc2007_labels

class TestDataLoader(unittest.TestCase):
    ROOT_DIR = "data/VOC2007"
    SPLIT = "train"
    data_loader = DataLoaderVOC2007(root_dir=ROOT_DIR, split=SPLIT)

    img, label = data_loader[3]
    plot_voc2007_labels(img, label)


if __name__ == "__main__":
    unittest.main()
