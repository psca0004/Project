

"""
Apple Detection Using RGB and Depth Images.

Identifies potential apples in a selected apple-tree image by
combining red/orange colour segmentation with depth filtering.
The dataset is read from a ZIP archive containing Summer and
Winter RGB and depth images.

Dataset:
    Source: appledata.zip
    Selected tree: R04U31
    Image categories: Summer RGB, Summer depth, Winter RGB,
        and Winter depth

Process:
    Counts the available images in each category and identifies
    the Summer RGB and matching Summer depth images for the
    selected tree.

    Extracts and loads the selected images, then creates a
    depth mask to retain pixels with grayscale values greater
    than or equal to 150.

    Converts the RGB image to HSV colour space and identifies
    red/orange regions using two colour masks.

    Combines the colour and depth masks, applies morphological
    opening and closing to reduce noise and fill small gaps,
    and identifies external contours in the resulting mask.

    Counts contours with an area greater than 100 pixels as
    potential apples.

Outputs:
    Prints the total number of unique trees and the number of
    images in each dataset category.

    Prints the selected tree ID and the number of potential
    apples detected.

Limitations:
    Detected contours are potential apples, not confirmed
    individual apples. Overlapping apples may be grouped into
    one contour, while other red/orange objects may be counted.
    Detection performance depends on the colour and depth
    thresholds and the image conditions.

Raises:
    StopIteration: If a matching Summer RGB or depth image
        cannot be found for the selected tree.
    FileNotFoundError: If either extracted image cannot be
        loaded successfully.
"""

import zipfile
import cv2
import numpy as np

data = "appledata.zip"
tree_id = "R04U31"

# Find images in the ZIP
with zipfile.ZipFile(data, "r") as zip_ref:
    files = [
        file for file in zip_ref.namelist()
        if file.lower().endswith(".png")
    ]

# Count the image categories
summer_rgb = [f for f in files if "_Summer_RGB" in f]
summer_depth = [f for f in files if "_Summer_D-" in f]
winter_rgb = [f for f in files if "_Winter_RGB" in f]
winter_depth = [f for f in files if "_Winter_D-" in f]

tree_ids = {
    f.split("/")[-1].split("_")[0]
    for f in files
}

print("Total trees:", len(tree_ids))
print("Summer RGB:", len(summer_rgb))
print("Summer Depth:", len(summer_depth))
print("Winter RGB:", len(winter_rgb))
print("Winter Depth:", len(winter_depth))

# Find the selected tree's images
rgb_file = next(
    f for f in summer_rgb
    if f.split("/")[-1].startswith(tree_id + "_")
)

depth_file = next(
    f for f in summer_depth
    if f.split("/")[-1].startswith(tree_id + "_")
)

# Extract the selected images
with zipfile.ZipFile(data, "r") as zip_ref:
    zip_ref.extract(rgb_file, "extracted")
    zip_ref.extract(depth_file, "extracted")

# Load RGB and depth
image = cv2.imread("extracted/" + rgb_file)
depth = cv2.imread(
    "extracted/" + depth_file,
    cv2.IMREAD_GRAYSCALE
)

if image is None or depth is None:
    raise FileNotFoundError("Could not load RGB or depth image")

# Keep close/light-grey depth pixels
close_threshold = 150
close_mask = (depth >= close_threshold).astype(np.uint8) * 255

# Convert RGB image to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Red/orange colour masks
mask1 = cv2.inRange(hsv, (0, 60, 50), (15, 255, 255))
mask2 = cv2.inRange(hsv, (165, 60, 50), (179, 255, 255))

mask = cv2.bitwise_or(mask1, mask2)

# Apply depth restriction
mask = cv2.bitwise_and(mask, mask, mask=close_mask)

# Remove noise and fill small gaps
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (5, 5)
)

mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find potential apples
contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

apple_count = 0

for contour in contours:
    if cv2.contourArea(contour) > 100:
        apple_count += 1

print("\nTree:", tree_id)
print("Potential apples detected:", apple_count)
