import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from tqdm import tqdm

# Define paths
dataset_root = "./Dataset"
train_csv_path = os.path.join(dataset_root, "train.csv")
train_images_dir = os.path.join(dataset_root, "train", "images")
train_labels_dir = os.path.join(dataset_root, "train", "labels")
output_images_dir = os.path.join(dataset_root, "train", "images")
output_labels_dir = os.path.join(dataset_root, "train", "labels")

# Create output directories if they don't exist
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Define the augmentation pipeline
augmentation_pipeline = A.Compose(
    [
        A.HorizontalFlip(p=0.5),  # Random horizontal flip
        A.RandomBrightnessContrast(p=0.5),  # Random brightness and contrast adjustment
        A.Rotate(limit=30, p=0.5),  # Random rotation within -30 to 30 degrees
        A.Blur(blur_limit=3, p=0.5),  # Random blur
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
)


# Function to load bounding boxes and class labels from a label file
def load_boxes_and_labels(label_path):
    boxes = []
    class_labels = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            boxes.append([x_center, y_center, width, height])
            class_labels.append(class_id)
    return np.array(boxes), class_labels


# Function to save bounding boxes and class labels to a label file
def save_boxes_and_labels(label_path, boxes, class_labels):
    with open(label_path, "w") as f:
        for box, class_id in zip(boxes, class_labels):
            x_center, y_center, width, height = box
            f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


# Load the training CSV file
train_df = pd.read_csv(train_csv_path)

# Create a list to store the new rows for the updated CSV
new_rows = []

# Process each image in the training set
for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Augmenting images"):
    image_path = os.path.join(train_images_dir, row["image"])
    label_path = os.path.join(train_labels_dir, row["label"])

    # Load the image and bounding boxes
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    boxes, class_labels = load_boxes_and_labels(label_path)

    # Generate 5 augmented images
    for i in range(5):
        # Apply augmentation
        augmented = augmentation_pipeline(image=image, bboxes=boxes, class_labels=class_labels)
        augmented_image = augmented["image"]
        augmented_boxes = augmented["bboxes"]
        augmented_class_labels = augmented["class_labels"]

        # Save the augmented image
        augmented_image_filename = f"{os.path.splitext(row['image'])[0]}_aug_{i}.jpg"
        augmented_image_path = os.path.join(output_images_dir, augmented_image_filename)
        cv2.imwrite(augmented_image_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

        # Save the augmented bounding boxes
        augmented_label_filename = f"{os.path.splitext(row['label'])[0]}_aug_{i}.txt"
        augmented_label_path = os.path.join(output_labels_dir, augmented_label_filename)
        save_boxes_and_labels(augmented_label_path, augmented_boxes, augmented_class_labels)

        # Add the new image and label to the list of new rows
        new_rows.append({"image": augmented_image_filename, "label": augmented_label_filename})

# Create a DataFrame from the new rows
augmented_df = pd.DataFrame(new_rows)

# Append the new rows to the original DataFrame
updated_df = pd.concat([train_df, augmented_df], ignore_index=True)

# Save the updated CSV file
updated_csv_path = os.path.join(dataset_root, "train.csv")
updated_df.to_csv(updated_csv_path, index=False)

print(f"Data augmentation complete! Updated CSV saved to {updated_csv_path}")
