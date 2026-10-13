
import zipfile
import cv2
import numpy as np


data = "appledata.zip"

with zipfile.ZipFile(data, "r") as zip_ref:
    files = zip_ref.namelist()

# Remove the folder name and look only at files
files = [file for file in files if file.lower().endswith(".png")]

summer_rgb = [file for file in files if "_Summer_RGB" in file]
summer_depth = [file for file in files if "_Summer_D-" in file]

winter_rgb = [file for file in files if "_Winter_RGB" in file]
winter_depth = [file for file in files if "_Winter_D-" in file]

# Get the tree ID from the beginning of each filename

tree_ids = []

for file in files:
    filename = file.split("/")[-1]
    tree_id = filename.split("_")[0]
    tree_ids.append(tree_id)

# Remove duplicates
tree_ids = list(set(tree_ids))


for tree_id in sorted(tree_ids)[:5]:

    tree_images = [file for file in files
                   if file.split("/")[-1].split("_")[0] == tree_id]

tree_id = "R04U31"

summer_depth_file = [
    file for file in files
    if file.split("/")[-1].startswith(tree_id + "_Summer_D")
][0]


# Find its Summer RGB image
summer_rgb_file = [
    file for file in files
    if file.split("/")[-1].startswith(tree_id + "_Summer_RGB")
][0]


# Extract that image
with zipfile.ZipFile(data, "r") as zip_ref:
    zip_ref.extract(summer_rgb_file, "extracted")
    zip_ref.extract(summer_depth_file, "extracted")


# Load image
image_path = "extracted/" + summer_rgb_file
image = cv2.imread(image_path)

depth_path = "extracted/" + summer_depth_file

depth = cv2.imread(
    depth_path,
    cv2.IMREAD_GRAYSCALE
)


# Create mask for close/light-grey areas
close_threshold = 150

close_mask = depth >= close_threshold

# Convert True/False into 255/0
close_mask = (
    close_mask.astype(np.uint8)
    * 255
)


# Detect apples using OpenCV

# Convert image from BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# Red/orange pixels
mask1 = cv2.inRange(
    hsv,
    (0, 60, 50),
    (15, 255, 255)
)


# Red pixels at the other end of the HSV scale
mask2 = cv2.inRange(
    hsv,
    (165, 60, 50),
    (179, 255, 255)
)


# Combine the two colour masks
mask = cv2.bitwise_or(mask1, mask2)


# ONLY keep red/orange pixels that are
# also in the close/light-grey depth areas
mask = cv2.bitwise_and(
    mask,
    mask,
    mask=close_mask
)


kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)


mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel
)


mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel
)


contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


apple_count = 0

for contour in contours:

    area = cv2.contourArea(contour)

    if area > 100:
        apple_count += 1


print("\nTree:", tree_id)
print("Potential apples detected:", apple_count)

#did conditon becuase touching apples are not being counted as 2

# Create folders for the machine learning dataset

import os

os.makedirs("apple_dataset/images/train", exist_ok=True)
os.makedirs("apple_dataset/images/val", exist_ok=True)

os.makedirs("apple_dataset/labels/train", exist_ok=True)
os.makedirs("apple_dataset/labels/val", exist_ok=True)

print("ML dataset folders created.")
# Step 2 - Extract 20 Summer RGB images for machine learning

import random
import os

# Get all Summer RGB images
summer_images = [
    file for file in files
    if "_Summer_RGB" in file
]

# Make the selection repeatable
random.seed(42)

# Shuffle the images
random.shuffle(summer_images)

# Select 20 images
selected_images = summer_images[:20]

# First 16 go into training
train_images = selected_images[:16]

# Last 4 go into validation
val_images = selected_images[16:]

# Extract training images
with zipfile.ZipFile(data, "r") as zip_ref:

    for file in train_images:
        filename = os.path.basename(file)

        zip_ref.extract(
            file,
            "temp_extract"
        )

        source = os.path.join("temp_extract", file)
        destination = os.path.join(
            "apple_dataset",
            "images",
            "train",
            filename
        )

        os.replace(source, destination)

    # Extract validation images
    for file in val_images:
        filename = os.path.basename(file)

        zip_ref.extract(
            file,
            "temp_extract"
        )

        source = os.path.join("temp_extract", file)
        destination = os.path.join(
            "apple_dataset",
            "images",
            "val",
            filename
        )

        os.replace(source, destination)


print("Training images:", len(train_images))
print("Validation images:", len(val_images))
print("Images extracted successfully.")

# Step 5 - Convert LabelMe JSON files to YOLO labels

import zipfile
import os

zip_file = "appledata.zip"
extract_folder = "extracted_full"

# Extract the entire dataset
print("Extracting dataset...")

with zipfile.ZipFile(zip_file, "r") as zip_ref:
    zip_ref.extractall(extract_folder)

print("Extraction complete.")

# Find all Summer RGB images
summer_rgb = []

for root, folders, files in os.walk(extract_folder):
    for file in files:
        if "_Summer_RGB" in file and file.lower().endswith(".png"):
            summer_rgb.append(os.path.join(root, file))

print("Summer RGB images found:", len(summer_rgb))

print("\nFirst 5:")
for image in summer_rgb[:5]:
    print(image)

from ultralytics import YOLO
import os
import csv

# Load your trained model
model = YOLO("runs/detect/train-6/weights/best.pt")

# Find all Summer RGB images
summer_rgb = []

for root, folders, files in os.walk("extracted_full"):
    for file in files:
        if "_Summer_RGB" in file and file.lower().endswith(".png"):
            summer_rgb.append(os.path.join(root, file))

# Sort so the images are processed in a consistent order
summer_rgb.sort()

print("Summer RGB images found:", len(summer_rgb))

# Store results
results_data = []

print("\nRunning apple detection...\n")

for i, image_path in enumerate(summer_rgb, start=1):

    # Run detection
    results = model.predict(
        source=image_path,
        conf=0.25,
        verbose=False
    )

    # Number of detected apples
    apple_count = len(results[0].boxes)

    # Get filename
    filename = os.path.basename(image_path)

    results_data.append([filename, apple_count])

    # Show progress
    print(f"{i}/544  {filename}  →  {apple_count} apples")

# Save results to CSV
csv_file = "summer_apple_counts.csv"

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "apple_count"])
    writer.writerows(results_data)

print("\nFinished!")
print("Results saved to:", csv_file)

# Basic summary
counts = [row[1] for row in results_data]

print("\nSummary:")
print("Images processed:", len(counts))
print("Total detected apples:", sum(counts))
print("Average apples per image:", sum(counts) / len(counts))
print("Minimum apples in an image:", min(counts))
print("Maximum apples in an image:", max(counts))