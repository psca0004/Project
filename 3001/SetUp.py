
import os
import random
import shutil

# Full apple dataset
source_folder = "extracted_full/Dataset"

# Previous annotated dataset
existing_folder = "AL2"

# Folder for new images
output_folder = "images3"
os.makedirs(output_folder, exist_ok=True)

# Find tree IDs already used in AL2
existing_ids = set()

for split in ["train", "valid", "test"]:
    folder = os.path.join(existing_folder, split)

    for root, dirs, files in os.walk(folder):
        for file in files:
            if "_Summer_RGB" in file:
                existing_ids.add(file.split("_")[0])

# Find available Summer RGB images
available_images = []

for file in os.listdir(source_folder):
    if "_Summer_RGB" in file and file.lower().endswith(".png"):
        tree_id = file.split("_")[0]

        if tree_id not in existing_ids:
            available_images.append(file)

# Select 43 new images
random.seed(43)
selected_images = random.sample(
    sorted(available_images), 43
)

# Copy selected images
for file in selected_images:
    shutil.copy2(
        os.path.join(source_folder, file),
        os.path.join(output_folder, file)
    )

print("New images selected:", len(selected_images))
print("Saved to:", output_folder)

for file in selected_images:
    print(file)
