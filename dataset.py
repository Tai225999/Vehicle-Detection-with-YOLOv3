import os
import cv2
import numpy as np
import torch
import config
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from utils import iou_width_height as iou
from pandas import read_csv

class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        C=7,
    ):
        """
        Args:
            csv_file (str): Path to the CSV file with image and label filenames.
            img_dir (str): Directory with all the images.
            label_dir (str): Directory with all the labels.
            anchors (list): List of anchor boxes for all scales.
            image_size (int): Size of the input image (assumed to be square).
            S (list): Grid sizes for each scale.
            C (int): Number of classes.
        """
        self.annotations = read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # Combine all anchors
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Load label path and bounding boxes
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2).tolist()  # Load labels directly

        # Load image using OpenCV
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = cv2.resize(image, (self.image_size, self.image_size))  # Resize to target size

        # Convert image to tensor and normalize to [0, 1]
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Build targets for each scale
        targets = [torch.zeros((self.num_anchors_per_scale, S, S, 6)) for S in self.S]
        for box in bboxes:
            class_id, x, y, width, height = box
            iou_anchors = iou(torch.tensor([width, height]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            has_anchor = [False] * 3  # Each scale should have one anchor

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # Which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # Both between [0, 1]
                    width_cell, height_cell = width * S, height * S  # Can be greater than 1
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_id)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # Ignore prediction

        return image, tuple(targets)