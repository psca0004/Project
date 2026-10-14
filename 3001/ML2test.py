# ============================================================
# MACHINE LEARNING MODEL 2 - TESTING
#
# PURPOSE:
# Test the second trained YOLO model on the same 16 Summer
# RGB images used to test Model 1.
#
# MODEL:
# - Dataset: AL2 (26 training images)
# - Confidence threshold: 0.25
# - Test images: 16
#
# OUTPUT:
# - Images showing detected apples and bounding boxes
# - CSV containing apple counts for each image
# - Summary of total and average detected apples
# ============================================================

import os
import csv
from ultralytics import YOLO

# Load the second trained YOLO model
model = YOLO(
    "runs/detect/runs/detect/apple_model_32/weights/best.pt"
)

# Read the same 16 images selected for Model 1
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
    name="apple_test_32"       # Results folder name
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
csv_file = "model2_16_apple_counts.csv"

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "apple_count"])
    writer.writerows(results_data)

# Calculate summary statistics
counts = [row[1] for row in results_data]

print("\nSummary:")
print("Images processed:", len(counts))
print("Total detected apples:", sum(counts))
print("Average apples per image:", sum(counts) / len(counts))
print("Minimum apples in an image:", min(counts))
print("Maximum apples in an image:", max(counts))

print("\nResults saved to:", csv_file)