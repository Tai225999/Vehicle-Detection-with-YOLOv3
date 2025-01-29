import os
import cv2
import numpy as np
import torch
import config
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from utils import iou_width_height as iou
from pandas import read_csv
from utils import iou_width_height

class YOLODataset(Dataset):
    def __init__(self,
                 csv_file,
                 img_dir,
                 label_dir,
                 anchors=torch.tensor(config.ANCHORS),
                 image_size=416,
                 S=[13, 26, 52],
                 C=7
                 # Since our dataset has already been preprocessed, we don't need to pass the transform argument
                 ):
        self.annotations = read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.S = S
        self.anchors = torch.cat([anchors[0], anchors[1], anchors[2]])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3 # Assume 3 scales
        self.C = C
        self.ignore_iou_threshold = 0.5

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        """
        Returns a tuple of (image, targets)
        image is an image tensor of shape (3, image_size, image_size)
        targets is a tensor of shape (3, S, S, num_anchors, 6)), where 3 is the number of scales
        The last dimension contains [objectness, x, y, w, h, class]
        """
        # Get the bounding boxes for this index
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2)
        bboxes = torch.tensor(bboxes, dtype=torch.float32)

        # Get the image for this index
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image / 255.0
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        # Adjust bounding box coordinates for resized image
        bboxes[:, 1:] = bboxes[:, 1:] * torch.tensor(
            [self.image_size, self.image_size, self.image_size, self.image_size]
        )

        # Build the target tensors
        targets = [torch.zeros((self.num_anchors_per_scale, self.S[i], self.S[i], 6)) for i in range(3)]

        # Extract box coordinates and class IDs
        class_ids = bboxes[:, 0].long()  # Shape: (N,)
        box_coords = bboxes[:, 1:]  # Shape: (N, 4)

        # Compute IoU between each box and all anchors
        iou_anchors = iou_width_height(box_coords[:, 2:] / torch.tensor([self.image_size, self.image_size]), self.anchors)  # Shape: (N, num_anchors)
        
        # Sort anchors by IoU in descending order
        anchor_indices = iou_anchors.argsort(dim=1, descending=True)  # Shape: (N, num_anchors)
        # Iterate over each scale
        for scale_idx in range(3):
            mask = (anchor_indices[:] >= self.num_anchors_per_scale * scale_idx) & (anchor_indices[:] < self.num_anchors_per_scale * (scale_idx + 1))
            anchor_indices_this_scale = anchor_indices[mask].reshape(anchor_indices.shape[0], -1)
            # Get grid size for this scale
            grid_size = self.S[scale_idx]

            # Compute cell indices (i, j) for each box
            i = (box_coords[:, 1] * grid_size / self.image_size).long()  # Shape: (N,)
            j = (box_coords[:, 0] * grid_size / self.image_size).long()  # Shape: (N,)

            # Compute cell-relative coordinates
            x_cell = (box_coords[:, 0] * grid_size / self.image_size) - j.float()  # Shape: (N,)
            y_cell = (box_coords[:, 1] * grid_size / self.image_size) - i.float()  # Shape: (N,)
            width_cell = (box_coords[:, 2] / self.image_size) * grid_size  # Shape: (N,)
            height_cell = (box_coords[:, 3] / self.image_size) * grid_size  # Shape: (N,)

            # Stack cell-relative coordinates into a single tensor
            box_coordinates = torch.stack([x_cell, y_cell, width_cell, height_cell], dim=1)  # Shape: (N, 4)

            # Iterate over each box
            for box_idx in range(bboxes.shape[0]):
                # Get the best anchor for this box
                best_anchor_idx = anchor_indices_this_scale[box_idx, 0] % self.num_anchors_per_scale

                # Check if the anchor is already taken
                if targets[scale_idx][best_anchor_idx, i[box_idx], j[box_idx], 0] == 0:
                    # Mark the anchor as taken
                    targets[scale_idx][best_anchor_idx, i[box_idx], j[box_idx], 0] = 1

                    # Assign box coordinates and class ID
                    targets[scale_idx][best_anchor_idx, i[box_idx], j[box_idx], 1:5] = box_coordinates[box_idx]
                    targets[scale_idx][best_anchor_idx, i[box_idx], j[box_idx], 5] = class_ids[box_idx]
                # Check for ignored predictions
                elif iou_anchors[box_idx, best_anchor_idx] > self.ignore_iou_threshold:
                    targets[scale_idx][best_anchor_idx, i[box_idx], j[box_idx], 0] = -1
        
        return image, tuple(targets)