import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET

VOC2007_LABELS = {
    "aeroplane": 0,
    "bicycle": 1,
    "boat": 2,
    "bottle": 3,
    "bus": 4,
    "car": 5,
    "cat": 6,
    "chair": 7,
    "diningtable": 8,
    "dog": 9,
    "horse": 10,
    "motorbike": 11,
    "pottedplant": 12,
    "sofa": 13,
    "train": 14,
    "tvmonitor": 15,
}

def voc2007_to_onehot(name):
    for 

class DataLoaderVOC2007(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = root_dir
        self.split_path = os.path.join(self.root_dir, "ImageSets", "Layout", split + ".txt")
        assert os.path.isfile(self.split_path), ("The split path is not correct: ", self.split_path)

        self.filenames = []

        # Read the split data distribution
        with open(self.split_path) as f:
            lines = f.readlines()
            for line in lines:
                image_filename = os.path.join(self.root_dir, "JPEGImages", line.strip() + ".jpg")
                annotation_filename = os.path.join(self.root_dir, "Annotations", line.strip() + ".xml")

                try:
                    assert os.path.isfile(image_filename)
                    assert os.path.isfile(annotation_filename)
                except:
                    print("File does not exist: ", image_filename, annotation_filename)

                dic_files = {"image": image_filename, "annotation": annotation_filename}
                self.filenames.append(dic_files)

    def parse_annotation(self, file_name):
        print("Parsing: ", file_name)
        root = ET.parse(file_name).getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text
            label = VOC2007_LABELS[name]
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            box = np.array((xmin, ymin, xmax, ymax))
            print(name, label, box)

    def __len__(self):
        return 0

    def __getitem__(self, index):
        return self.filenames[index]
