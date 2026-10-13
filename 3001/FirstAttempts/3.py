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

