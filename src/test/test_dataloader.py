import unittest
from src.dataloader import DataLoaderVOC2007
from src.utils.plot import plot_voc2007_boxes, plot_voc2007_labels
from src.utils.utils import resize_data, convert_annotation_to_label

class TestDataLoader(unittest.TestCase):
    ROOT_DIR = "data/VOC2007"
    SPLIT = "train"
    data_loader = DataLoaderVOC2007(ROOT_DIR, SPLIT)

    K = 71
    image_filename = data_loader[K]["image"]
    img_size, annotations = data_loader.parse_annotation(data_loader[K]["annotation"])

    img, annotations = resize_data(image_filename, annotations)
    # plot_voc2007_boxes(img, annotations)
    annotations = convert_annotation_to_label(annotations)
    plot_voc2007_labels(img, annotations)
    

if __name__ == "__main__":
    unittest.main()
