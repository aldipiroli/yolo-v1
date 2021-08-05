import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import matplotlib.image as mpimg
from skimage import io, transform

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


def label_to_onehot(name, C=20):
    one_hot_label = np.zeros((C))
    idx = VOC2007_LABELS[name]
    one_hot_label[idx] = 1
    return one_hot_label.reshape(-1, 20)


class DataLoaderVOC2007(Dataset):
    def __init__(self, root_dir, S=7, H=448, W=448, C=20, B=2, split="train"):
        self.S = S
        self.W = W
        self.H = H
        self.C = C
        self.B = B
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

        annotations = []
        for obj in root.findall("object"):

            name = obj.find("name").text
            label = VOC2007_LABELS[name]
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            box = np.array((xmin, ymin, xmax, ymax)).reshape(-1, 4)
            label_one_hot = label_to_onehot(name)

            ann = np.concatenate((box, label_one_hot), axis=1)
            annotations.append(ann.squeeze())

        # Get image size:
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        depth = int(size.find("depth").text)

        img_size = np.array((width, height, depth))

        return annotations

    def resize_data(self, img, annotations):
        """
        Resize an image and relative annotation to a desired output
        """
        h, w = img.shape[0], img.shape[1]

        new_h, new_w = self.H, self.W
        new_img = transform.resize(img, (new_h, new_w))

        ann_ratio = np.array([new_w / w, new_h / h])

        for ann in annotations:
            ann[0] = np.clip(ann[0] * ann_ratio[0], 0, new_w)
            ann[1] = np.clip(ann[1] * ann_ratio[1], 0, new_h)
            ann[2] = np.clip(ann[2] * ann_ratio[0], 0, new_w)
            ann[3] = np.clip(ann[3] * ann_ratio[1], 0, new_w)

        return new_img, annotations

    def load_img(self, image_filename):
        img = mpimg.imread(image_filename)
        return img

    def convert_coordinate_abs_rel(self, center_x, center_y, box):
        """Convert the coordinates from absolute value (image) to relative one of the bbox"""
        new_x = center_x - box[0]
        new_y = center_y - box[2]

        return new_x, new_y

    def is_inside_box(self, center_x, center_y, box):
        if center_x > box[0] and center_x < box[1]:
            if center_y > box[2] and center_y < box[3]:
                return True

        return False

    def convert_annotation_to_label(self, annotations):
        label = np.zeros((self.S, self.S, 4 + self.C))
        STEP = self.H / 7

        for i in range(self.S):
            for j in range(self.S):
                x_min, x_max = STEP * i, STEP * (i + 1)
                y_min, y_max = STEP * j, STEP * (j + 1)

                box = np.array((x_min, x_max, y_min, y_max))

                for k in range(len(annotations)):
                    ann = annotations[k]
                    ann_x_min, ann_y_min = ann[0], ann[1]
                    ann_x_max, ann_y_max = ann[2], ann[3]
                    center_x = (ann_x_max - ann_x_min) / 2 + ann_x_min
                    center_y = (ann_y_max - ann_y_min) / 2 + ann_y_min

                    if self.is_inside_box(center_x, center_y, box):
                        x, y = self.convert_coordinate_abs_rel(center_x, center_y, box)
                        w = ann_x_max - ann_x_min
                        h = ann_y_max - ann_y_min
                        label[i, j, :4] = [x, y, w, h]
                        label[i, j, 4:] = ann[4:]

        return label

    def __getitem__(self, index):
        image_filename = self.filenames[index]["image"]
        annotation_filename = self.filenames[index]["annotation"]

        img = self.load_img(image_filename)
        annotations = self.parse_annotation(annotation_filename)

        img, annotations = self.resize_data(img, annotations)
        annotations = self.convert_annotation_to_label(annotations)

        return img, annotations

    def __len__(self, idx):
        return len(self.filenames)
