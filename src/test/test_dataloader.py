import unittest
from src.dataloader import DataLoaderVOC2007
from src.utils.plot import plot_voc2007_full_boxes, resize_data

class TestDataLoader(unittest.TestCase):
    ROOT_DIR = "/home/aldi/workspace/yolo-v1/data/VOC2007"
    SPLIT = "train"
    data_loader = DataLoaderVOC2007(ROOT_DIR, SPLIT)

    K = 72
    image_filename = data_loader[K]["image"]
    img_size, annotations = data_loader.parse_annotation(data_loader[K]["annotation"])

    img, annotations = resize_data(image_filename, annotations)
    plot_voc2007_full_boxes(img, annotations)
    

if __name__ == "__main__":
    unittest.main()
