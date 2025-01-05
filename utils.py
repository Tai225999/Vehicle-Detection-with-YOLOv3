import torch
import cv2

import config

def iou_width_height(bbox1, bbox2):
    """
    Calculate intersection over union for the width and height of two bounding boxes (no regard for the coordinates).
    Args:
        bbox1(torch.Tensor): bounding box 1, a tensor with width and height
        bbox2(torch.Tensor): bounding box 2, a tensor with width and height
    Returns:
        torch.Tensor: intersection over union for width and height
    """

    width_height1 = bbox1[0] * bbox1[1]
    width_height2 = bbox2[0] * bbox2[1]

    intersection = torch.min(bbox1, bbox2).prod()
    union = width_height1 + width_height2 - intersection + 1e-6

    return intersection / union


def intersection_over_union(bbox1, bbox2):
    """
    Calculate intersection over union for two batches of bounding boxes.
    Args:
        bbox1 (torch.Tensor): bounding box 1, a tensor of shape (N, 4).
        bbox2 (torch.Tensor): bounding box 2, a tensor of shape (N, 4).
    Returns:
        torch.Tensor: intersection over union, a tensor of shape (N,).
    """

    # Get the coordinates of bounding boxes
    x1, y1, w1, h1 = bbox1[..., 0:1], bbox1[..., 1:2], bbox1[..., 2:3], bbox1[..., 3:4]
    x2, y2, w2, h2 = bbox2[..., 0:1], bbox2[..., 1:2], bbox2[..., 2:3], bbox2[..., 3:4]

    # Get the coordinates of the intersection rectangle
    x1_intersection = torch.max(x1 - w1 / 2, x2 - w2 / 2)
    y1_intersection = torch.max(y1 - h1 / 2, y2 - h2 / 2)
    x2_intersection = torch.min(x1 + w1 / 2, x2 + w2 / 2)
    y2_intersection = torch.min(y1 + h1 / 2, y2 + h2 / 2)

    # Get the area of intersection rectangle
    intersection = (x2_intersection - x1_intersection).clamp(0) * (y2_intersection - y1_intersection).clamp(0)

    # Get the area of both bounding boxes
    area1 = w1 * h1
    area2 = w2 * h2

    # Get the area of union
    union = area1 + area2 - intersection + 1e-6 # 1e-6 is used to prevent division by zero

    return intersection / union


def non_max_suppression(bboxes, iou_threshold, threshold):
    """
    Perform non-maximum suppression on the bounding boxes.
    Args:
        bboxes (list): Bounding boxes to perform NMS on. Each bounding box is in the format [class, score, x, y, w, h].
        iou_threshold (float): IoU threshold for overlapping boxes.
        threshold (float): Threshold for removing boxes.
    Returns:
        list: Remaining boxes after NMS.
    """

    # If no bounding boxes, return an empty tensor
    if len(bboxes) == 0:
        return torch.tensor([])

    # Eliminate boxes with scores lower than the threshold
    bboxes = [box for box in bboxes if box[1] > threshold]

    # Sort the bounding boxes by their scores
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    # List to hold the best bounding boxes after NMS
    best_bboxes = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        # Append the chosen bounding box
        best_bboxes.append(chosen_box)

        # Update bbox, only keep the boxes with IoU less than the threshold and the class index is different
        bboxes = [box for box in bboxes
                  if box[0] != chosen_box[0]
                  or intersection_over_union(torch.tensor(chosen_box[2:]), torch.tensor(box[2:])) < iou_threshold]
        
    return best_bboxes

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    """
    Save model checkpoint.
    Args:
        model: Model to save.
        optimizer: Optimizer of the model.
        filename (str): Name of the file.
    """
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    """
    Load model checkpoint.
    Args:
        model: Model to load the checkpoint to.
        optimizer: Optimizer of the model.
        filename (str): Name of the file.
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"Checkpoint loaded from {filename}")


def show_image(image, boxes, predict=False):
    """
    Show image with bounding boxes.
    Args:
        image (torch.Tensor): Image tensor of shape (C, H, W).
        boxes (torch.Tensor): Bounding boxes.
                             If predict is False, it takes the form of a tensor with shape (N, 5),
                             where the latter dimension represents (class, x, y, w, h).
                             If predict is True, it is not yet implemented.
        predict (bool): Whether the boxes are predictions (True) or ground truths (False).
    """
    classes = config.CLASSES  # List of class names

    if predict:
        raise NotImplementedError("This function is not yet implemented for predictions.")
    else:
        # Convert image tensor to numpy array and permute dimensions
        image = image.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

        # Get image dimensions
        image_height, image_width, _ = image.shape

        for box in boxes:
            # Extract box coordinates and class index
            class_idx = int(box[0].item())  # Convert tensor to int
            x, y, w, h = box[1:].tolist()  # Convert tensor to list

            # Scale coordinates if normalized
            if x < 1.0 and y < 1.0 and w < 1.0 and h < 1.0:
                x = int(x * image_width)
                y = int(y * image_height)
                w = int(w * image_width)
                h = int(h * image_height)

            # Draw bounding box
            color = (255, 0, 0)  # Red color
            thickness = 2
            cv2.rectangle(image, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), color, thickness)

            # Add class label
            class_name = classes[class_idx]
            cv2.putText(image, class_name, (int(x - w / 2), int(y - h / 2 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)

        # Display the image
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
