import os
import torch
import pandas as pd
from tqdm import tqdm
from utils import iou_width_height


def kmeans_iou(boxes, k, max_iters=30):
    """
    Perform k-means clustering using IoU as the distance metric with PyTorch.
    Args:
        boxes: Tensor of bounding boxes (width, height) of shape (N, 2).
        k: Number of clusters.
        max_iters: Maximum number of iterations.
    Returns:
        centroids: The final cluster centers (anchor boxes) as a PyTorch tensor.
    """
    # Ensure boxes is a PyTorch tensor
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes, dtype=torch.float32)

    # Initialize centroids randomly
    torch.manual_seed(42)  # For reproducibility
    indices = torch.randperm(boxes.size(0))[:k]
    centroids = boxes[indices]

    for _ in tqdm(range(max_iters), desc="K-means clustering"):
        # Compute pairwise IoU distances between boxes and centroids
        distances = 1 - torch.stack([
            torch.tensor([iou_width_height(box, centroid) for centroid in centroids])
            for box in boxes
        ])

        # Assign each box to the closest centroid
        labels = torch.argmin(distances, dim=1)

        # Update centroids
        new_centroids = torch.stack([
            boxes[labels == i].mean(dim=0) for i in range(k)
        ])

        # Print the centroids for each iteration
        print(f'Centroids: {new_centroids}')

        # Check for convergence
        if torch.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return centroids


def load_wh_from_labels(csv_path, labels_dir):
    """
    Load widths and heights of bounding boxes from label files.
    Args:
        csv_path: Path to the CSV file mapping images to labels.
        labels_dir: Directory containing label files.
    Returns:
        widths_heights: List of (width, height) tuples.
    """
    df = pd.read_csv(csv_path)
    widths_heights = []

    for _, row in df.iterrows():
        label_path = os.path.join(labels_dir, row["label"])
        with open(label_path, "r") as f:
            for line in f.readlines():
                _, _, _, w, h = map(float, line.strip().split())
                widths_heights.append([w, h])

    return torch.tensor(widths_heights, dtype=torch.float32)  # Convert to PyTorch tensor


def generate_anchor_boxes(dataset_root, num_anchors=9):
    """
    Generate anchor boxes using k-means clustering with IoU as the distance metric.
    Args:
        dataset_root: Root directory of the dataset.
        num_anchors: Total number of anchor boxes to generate.
    Returns:
        anchors: The generated anchor boxes.
    """
    # Load widths and heights from all splits (train, valid, test)
    train_csv = os.path.join(dataset_root, "train.csv")
    valid_csv = os.path.join(dataset_root, "valid.csv")
    test_csv = os.path.join(dataset_root, "test.csv")

    train_labels_dir = os.path.join(dataset_root, "train", "labels")
    valid_labels_dir = os.path.join(dataset_root, "valid", "labels")
    test_labels_dir = os.path.join(dataset_root, "test", "labels")

    train_wh = load_wh_from_labels(train_csv, train_labels_dir)
    valid_wh = load_wh_from_labels(valid_csv, valid_labels_dir)
    test_wh = load_wh_from_labels(test_csv, test_labels_dir)

    all_wh = torch.cat([train_wh, valid_wh, test_wh], dim=0)  # Concatenate tensors

    # Run k-means clustering with IoU as the distance metric
    anchors = kmeans_iou(all_wh, k=num_anchors)

    # Sort anchors by area (optional but recommended for YOLO)
    anchors = sorted(anchors, key=lambda x: x[0] * x[1], reverse=True)

    return anchors


if __name__ == "__main__":
    # Path to the dataset root directory
    dataset_root = "./Dataset"

    # Generate anchor boxes
    num_anchors = 9  # Total number of anchor boxes
    anchors = generate_anchor_boxes(dataset_root, num_anchors)

    # Print the anchor boxes
    print("Anchor Boxes (width, height):")
    for anchor in anchors:
        print(f"({anchor[0]:.2f}, {anchor[1]:.2f})")

    # Save the anchors to a file (optional)
    with open(os.path.join(dataset_root, "anchors.txt"), "w") as f:
        for anchor in anchors:
            f.write(f"{anchor[0]:.2f},{anchor[1]:.2f}\n")
