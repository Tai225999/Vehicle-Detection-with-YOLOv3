import torch

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