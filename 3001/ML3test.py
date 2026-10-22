
"""
Machine Learning Model 3 - Testing.

Tests the third trained YOLO model on the same 16 Summer RGB
images used to evaluate Models 1 and 2. Detects apples and
records the number of detections in each image to allow
consistent comparison between the three models.

Dataset:
    Training dataset: AL3 (56 training images)
    Test images: 16 Summer RGB images

Model Configuration:
    Model: Trained YOLO11n
    Confidence threshold: 0.25

Process:
    Loads the trained model and reads the selected image
    filenames from 'comparison_16_images.csv'.

    Checks that each selected image exists and that exactly
    16 images have been selected.

    Runs apple detection on each image and records the
    number of detected apples.

Outputs:
    Saves annotated images showing detected apples and
    their bounding boxes under 'runs/detect/apple_test_64'.

    Exports apple counts for each image to
    'model3_16_apple_counts.csv'.

    Displays summary statistics including the total, average,
    minimum, and maximum number of detected apples per image.

Raises:
    FileNotFoundError: If a selected image does not exist.
    ValueError: If the selection does not contain exactly
        16 images.
"""

import os
import csv
from ultralytics import YOLO

# Load the third trained YOLO model
model = YOLO(
    "runs/detect/runs/apple_model_64-9/weights/best.pt"
)

# Read the same 16 images selected for Models 1 and 2
selection_file = "comparison_16_images.csv"
image_folder = "extracted_full/Dataset"

selected_images = []

with open(selection_file, "r", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        image_path = os.path.join(image_folder, row["image"])

        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)

        selected_images.append(image_path)

# Check that exactly 16 images were selected
if len(selected_images) != 16:
    raise ValueError("Expected exactly 16 comparison images")

print("Images selected:", len(selected_images))

# Run apple detection on the 16 images
results = model.predict(
    source=selected_images,
    conf=0.25,                 # Minimum detection confidence
    save=True,                 # Save images with bounding boxes
    project="runs/detect",     # Results directory
    name="apple_test_64"       # Results folder name
)

# Store apple counts for each image
results_data = []

print("\nDetection results:")

for result in results:

    apple_count = len(result.boxes)
    filename = os.path.basename(result.path)

    results_data.append([filename, apple_count])

    print(filename, "->", apple_count, "apples detected")

# Save detection counts to CSV
csv_file = "model3_16_apple_counts.csv"

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "apple_count"])
    writer.writerows(results_data)

# Calculate summary statistics
counts = [row[1] for row in results_data]

print("\nSummary:")
print("Images processed:", len(counts))
print("Total apples detected:", sum(counts))
print("Average apples per image:", sum(counts) / len(counts))
print("Minimum apples in an image:", min(counts))
print("Maximum apples in an image:", max(counts))

print("\nResults saved to:", csv_file)
