

import zipfile
import os
import csv
from ultralytics import YOLO

# Dataset locations
zip_file = "appledata.zip"
extract_folder = "extracted_full"

# First trained YOLO model
model_path = "runs/detect/train-6/weights/best.pt"

# Extract full dataset if not already extracted
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

if not summer_rgb:
    raise RuntimeError("No Summer RGB images found")

# Load trained model
model = YOLO(model_path)

# Store results
results_data = []

print("\nRunning apple detection...\n")

for i, image_path in enumerate(summer_rgb, start=1):

    results = model.predict(
        source=image_path,
        conf=0.25,
        verbose=False
    )

    apple_count = len(results[0].boxes)
    filename = os.path.basename(image_path)

    results_data.append([filename, apple_count])

    print(
        f"{i}/{len(summer_rgb)} "
        f"{filename} -> {apple_count} apples"
    )

# Save results
csv_file = "summer_apple_counts.csv"

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "apple_count"])
    writer.writerows(results_data)

# Summary
counts = [row[1] for row in results_data]

print("\nFinished!")
print("Results saved to:", csv_file)

print("\nSummary:")
print("Images processed:", len(counts))
print("Total detected apples:", sum(counts))
print("Average apples per image:", sum(counts) / len(counts))
print("Minimum apples in an image:", min(counts))
print("Maximum apples in an image:", max(counts))
