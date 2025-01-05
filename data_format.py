import os
import cv2
import pandas as pd
from tqdm import tqdm

# Define paths
dataset_root = "./Dataset"
sets = ["train", "valid", "test"]


# Function to rename, move, and resize files
def process_dataset(input_set_name):
    # Define paths for the current set
    set_path = os.path.join(dataset_root, input_set_name)
    image_dir = os.path.join(set_path, "images")
    label_dir = os.path.join(set_path, "labels")

    # Create images and labels directories if they don't exist
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # Get all image and label files
    image_files = [f for f in os.listdir(set_path) if f.endswith(".jpg")]
    label_files = [f for f in os.listdir(set_path) if f.endswith(".txt")]

    # Sort files to ensure consistent ordering
    image_files.sort()
    label_files.sort()

    # Initialize list to store image-label mappings
    mappings = []

    # Process each image and label pair
    for idx, (image_file, label_file) in enumerate(tqdm(zip(image_files, label_files), desc=f"Processing {input_set_name}")):
        # Generate new filenames
        new_filename = f"{idx + 1:04d}"  # 0001, 0002, etc.
        new_image_filename = f"{new_filename}.jpg"
        new_label_filename = f"{new_filename}.txt"

        # Define full paths for old and new files
        old_image_path = os.path.join(set_path, image_file)
        old_label_path = os.path.join(set_path, label_file)
        new_image_path = os.path.join(image_dir, new_image_filename)
        new_label_path = os.path.join(label_dir, new_label_filename)

        # Resize image to 416x416
        image = cv2.imread(old_image_path)
        resized_image = cv2.resize(image, (416, 416))
        cv2.imwrite(new_image_path, resized_image)

        # Move label file
        os.rename(old_label_path, new_label_path)

        # Add mapping to the list
        mappings.append({"image": new_image_filename, "label": new_label_filename})

    # Save mappings to CSV
    df = pd.DataFrame(mappings)
    df.to_csv(os.path.join(dataset_root, f"{input_set_name}.csv"), index=False)


# Process each set (train, valid, test)
for set_name in sets:
    process_dataset(set_name)

print("Dataset processing complete!")
