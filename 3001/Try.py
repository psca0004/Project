import os
import json
import shutil
from collections import defaultdict

dataset = "AL2"

for split in ["train", "valid", "test"]:

    folder = os.path.join(dataset, split)
    json_path = os.path.join(folder, f"{split}.coco.json")

    with open(json_path, "r") as f:
        coco = json.load(f)

    images_folder = os.path.join(folder, "images")
    labels_folder = os.path.join(folder, "labels")

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    apple_ids = {
        c["id"] for c in coco["categories"]
        if c["name"].lower() in ["apple", "apples"]
    }

    annotations = defaultdict(list)

    for ann in coco["annotations"]:
        if ann["category_id"] in apple_ids:
            annotations[ann["image_id"]].append(ann)

    total_apples = 0

    for img in coco["images"]:

        filename = img["file_name"]
        width = img["width"]
        height = img["height"]

        source = os.path.join(folder, filename)
        destination = os.path.join(images_folder, filename)

        if not os.path.exists(source):
            raise FileNotFoundError(source)

        shutil.copy2(source, destination)

        label_path = os.path.join(
            labels_folder,
            os.path.splitext(filename)[0] + ".txt"
        )

        with open(label_path, "w") as f:

            for ann in annotations[img["id"]]:

                x, y, w, h = ann["bbox"]

                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height

                w = w / width
                h = h / height

                f.write(
                    f"0 {x_center:.6f} {y_center:.6f} "
                    f"{w:.6f} {h:.6f}\n"
                )

                total_apples += 1

    print(
        f"{split}: {len(coco['images'])} images, "
        f"{total_apples} apples"
    )

print("\nConversion complete!")