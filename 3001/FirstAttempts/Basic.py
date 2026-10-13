
import zipfile
import cv2



data = "appledata.zip"

with zipfile.ZipFile(data, "r") as zip_ref:
    files = zip_ref.namelist()

# Remove the folder name and look only at files
files = [file for file in files if file.lower().endswith(".png")]

summer_rgb = [file for file in files if "_Summer_RGB" in file]
summer_depth = [file for file in files if "_Summer_D-" in file]

winter_rgb = [file for file in files if "_Winter_RGB" in file]
winter_depth = [file for file in files if "_Winter_D-" in file]

print("Total PNG files:", len(files))

print("\nSummer RGB:", len(summer_rgb))
print("Summer Depth:", len(summer_depth))

print("\nWinter RGB:", len(winter_rgb))
print("Winter Depth:", len(winter_depth))
# Get the tree ID from the beginning of each filename

tree_ids = []

for file in files:
    filename = file.split("/")[-1]
    tree_id = filename.split("_")[0]
    tree_ids.append(tree_id)

# Remove duplicates
tree_ids = list(set(tree_ids))

print("Number of unique trees:", len(tree_ids))

for tree_id in sorted(tree_ids)[:5]:

    tree_images = [file for file in files
                   if file.split("/")[-1].split("_")[0] == tree_id]

tree_id = "R01U01"

# Find its Summer RGB image 
summer_rgb_file = [file for file in files if file.split("/")[-1].startswith(tree_id + "_Summer_RGB")][0] 
# Extract that image 
with zipfile.ZipFile(data, "r") as zip_ref: zip_ref.extract(summer_rgb_file, "extracted") 
# Load image 
image_path = "extracted/" + summer_rgb_file 
image = cv2.imread(image_path) 
# Detect apples using OpenCV

# Convert image from BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define a colour range for red apples
lower_red = (40, 50, 120)
upper_red = (170, 140, 255)

# Create a mask for pixels in this colour range
mask = cv2.inRange(hsv, lower_red, upper_red)

# Find connected regions
contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Count potential apples
apple_count = 0

for contour in contours:

    area = cv2.contourArea(contour)

    # Ignore very small regions
    if area > 100:
        apple_count += 1

print("\nTree:", tree_id)
print("Potential apples detected:", apple_count)

# Im just detecing specific images to see if it can count apples.
# Im struggling to pinpoint range of rgb. 
# I think is where machine learning comes in. 