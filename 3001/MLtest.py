
"""
Machine Learning Model 1 - Testing.

Tests the first trained YOLO model on 16 Summer RGB images
to detect and count apples. The same images are used to
evaluate Models 1, 2, and 3 for consistent comparison.

Dataset:
    Source: appledata.zip
    Training dataset: AL (11 training images)
    Test images: 16 Summer RGB images

Model Configuration:
    Model: Trained YOLO11n
    Confidence threshold: 0.25
    Random selection seed: 42

Process:
    Extracts the image dataset if necessary and identifies
    all available Summer RGB images.

    Selects 16 random images using a fixed seed, or reuses
    a previously saved selection to ensure consistent
    comparison between models.

    Performs apple detection on each selected image and
    records the number of detected apples.

Outputs:
    Saves selected image filenames to
    'comparison_16_images.csv' if not already present.

    Exports detected apple counts to
    'model1_16_apple_counts.csv'.

    Displays the total, average, minimum, and maximum
    number of detected apples per image.

Raises:
    RuntimeError: If fewer than 16 Summer RGB images are found.
    KeyError: If a previously selected image is not found
        in the extracted dataset.
"""

import zipfile
import os
import csv
import random
from ultralytics import YOLO

# Dataset locations
zip_file = "appledata.zip"
extract_folder = "extracted_full"

# First trained YOLO model
model_path = "runs/detect/train-6/weights/best.pt"

# File storing the 16 selected images
selection_file = "comparison_16_images.csv"

# Extract the dataset if necessary
dataset_folder = os.path.join(extract_folder, "Dataset")

if not os.path.isdir(dataset_folder):
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

summer_rgb.sort()

print("Summer RGB images found:", len(summer_rgb))

if len(summer_rgb) < 16:
    raise RuntimeError("Not enough Summer RGB images found")

# ============================================================
# SELECT 16 IMAGES FOR MODEL COMPARISON
# ============================================================

# Reuse the same images if a selection already exists
if os.path.exists(selection_file):

    with open(selection_file, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        selected_names = [row[0] for row in reader]

    image_lookup = {
        os.path.basename(path): path for path in summer_rgb
    }

    selected_images = [image_lookup[name] for name in selected_names]

else:

    # Select 16 random images with a fixed seed
    random.seed(42)
    selected_images = random.sample(summer_rgb, 16)

    # Save selection for future models
    with open(selection_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image"])

        for path in selected_images:
            writer.writerow([os.path.basename(path)])

print("Images selected:", len(selected_images))

# ============================================================
# RUN APPLE DETECTION
# ============================================================

# Load trained model
model = YOLO(model_path)

# Store detection results
results_data = []

print("\nRunning apple detection...\n")

for i, image_path in enumerate(selected_images, start=1):

    # Detect apples in the image
    results = model.predict(
        source=image_path,
        conf=0.25,
        verbose=False
    )

    # Count detected apples
    apple_count = len(results[0].boxes)
    filename = os.path.basename(image_path)

    results_data.append([filename, apple_count])

    print(
        f"{i}/{len(selected_images)} "
        f"{filename} -> {apple_count} apples"
    )

# ============================================================
# SAVE RESULTS
# ============================================================

# Save Model 1 results to CSV
csv_file = "model1_16_apple_counts.csv"

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "apple_count"])
    writer.writerows(results_data)

# ============================================================
# DISPLAY SUMMARY
# ============================================================

counts = [row[1] for row in results_data]

print("\nFinished!")
print("Results saved to:", csv_file)

print("\nSummary:")
print("Images processed:", len(counts))
print("Total detected apples:", sum(counts))
print("Average apples per image:", sum(counts) / len(counts))
print("Minimum apples in an image:", min(counts))
print("Maximum apples in an image:", max(counts))
