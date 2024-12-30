import json
import pandas as pd
import os
import random
from shutil import copy2
def generate_yolo_labels(image_folder, label_folder, info_indexed):
    """
    Generate YOLO-compatible .txt labels for the given folder.
    """
    os.makedirs(label_folder, exist_ok=True)
    file_names = os.listdir(image_folder)

    for f in file_names:
        try:
            # Get normalized bounding box info
            bbox_norm = info_indexed.loc[f, 'bbox_normalized']
            bbox_norm_str = " ".join(map(str, bbox_norm))

            # Create corresponding .txt file
            txt_file_path = os.path.join(label_folder, f.split('.')[0] + '.txt')
            with open(txt_file_path, "w") as file:
                file.write(f"0 {bbox_norm_str}")

            print(f"Label file created: {txt_file_path}")

        except KeyError:
            print(f"File {f} not found in info_indexed!")

# Load JSON file
with open("data/probe_dataset/probe_labels.json", "r") as file:
    data = json.load(file)

# Create DataFrames for annotations and images
anno = pd.DataFrame(data['annotations']).set_index('image_id')
image = pd.DataFrame(data['images']).set_index('id')

# Merge data on image_id and prepare for indexing by file_name
info = pd.merge(image, anno, left_on='id', right_on='image_id')

# Display unique counts for height and width
print('Unique heights:', info['height'].nunique())
print('Unique widths:', info['width'].nunique())

# Normalize bounding boxes
info['bbox_normalized'] = info.apply(
    lambda row: [
        (row['bbox'][0] + row['bbox'][2] / 2) / row['width'],
        (row['bbox'][1] + row['bbox'][3] / 2) / row['height'],
        row['bbox'][2] / row['width'],
        row['bbox'][3] / row['height']
    ],
    axis=1
)
info_indexed = info.set_index('file_name')
# Define dataset paths and train/val split ratio
dataset_path = "data/probe_dataset/probe_images"
output_path = "data/yolov11/images"
train_ratio = 0.8

# Create directories for train and validation sets
os.makedirs(f"{output_path}/train", exist_ok=True)
os.makedirs(f"{output_path}/val", exist_ok=True)

# Shuffle and split files
files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
random.shuffle(files)
train_count = int(len(files) * train_ratio)

# Split into train and val sets
train_files = files[:train_count]
val_files = files[train_count:]

# Copy files to respective directories
for f in train_files:
    copy2(os.path.join(dataset_path, f), os.path.join(output_path, "train"))
for f in val_files:
    copy2(os.path.join(dataset_path, f), os.path.join(output_path, "val"))

print("Dataset split completed!")


generate_yolo_labels(
    image_folder=f"{output_path}/train",
    label_folder="data/yolov11/labels/train",
    info_indexed=info_indexed
)

generate_yolo_labels(
    image_folder=f"{output_path}/val",
    label_folder="data/yolov11/labels/val",
    info_indexed=info_indexed
)

csv_output_path = "data/info_indexed.csv"
info_indexed.to_csv(csv_output_path, index=True)