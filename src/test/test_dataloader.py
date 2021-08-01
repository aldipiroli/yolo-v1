import unittest
from src.dataloader import DataLoaderVOC2007

class TestDataLoader(unittest.TestCase):
    ROOT_DIR = "/home/aldi/workspace/yolo-v1/data/VOC2007"
    SPLIT = "train"
    data_loader = DataLoaderVOC2007(ROOT_DIR, SPLIT)

    data_loader.parse_annotation(data_loader[2]["annotation"])

if __name__ == "__main__":
    unittest.main()
