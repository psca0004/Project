
import json
import os
import glob
import cv2

# Validation folders
image_folder = "apple_dataset/images/val"
label_folder = "apple_dataset/labels/val"

os.makedirs(label_folder, exist_ok=True)

# Find validation JSON files
json_files = glob.glob(os.path.join(image_folder, "*.json"))

print("JSON files found:", len(json_files))

for json_file in json_files:

    with open(json_file, "r") as f:
        data_json = json.load(f)

    # Matching PNG image
    image_name = os.path.splitext(os.path.basename(json_file))[0] + ".png"
    image_path = os.path.join(image_folder, image_name)

    image = cv2.imread(image_path)

    if image is None:
        print("Could not find image:", image_name)
        continue

    height, width = image.shape[:2]

    # Output YOLO label file
    label_name = os.path.splitext(os.path.basename(json_file))[0] + ".txt"
    label_path = os.path.join(label_folder, label_name)

    yolo_lines = []

    # Go through every labelled shape
    for shape in data_json["shapes"]:

        if shape["label"].lower() != "apple":
            continue

        points = shape["points"]

        # Find bounding box around polygon
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]

        x_min = min(x_values)
        x_max = max(x_values)
        y_min = min(y_values)
        y_max = max(y_values)

        # Bounding box centre
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        # Bounding box size
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Convert to YOLO's 0–1 format
        x_center /= width
        y_center /= height
        box_width /= width
        box_height /= height

        # Class 0 = Apple
        yolo_line = (
            f"0 "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )

        yolo_lines.append(yolo_line)

    # Save YOLO file
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines))

    print(
        "Converted:",
        os.path.basename(json_file),
        "→",
        os.path.basename(label_path),
        "Apples:",
        len(yolo_lines)
    )

print("\nValidation conversion complete!")