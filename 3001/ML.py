

from ultralytics import YOLO

# First machine learning dataset: AL
# 11 training images, 3 validation images, 2 test images

# Load pretrained YOLO11 nano model
model = YOLO("yolo11n.pt")

# Train on the first dataset
model.train(
    data="AL/apple.yaml",
    epochs=100,
    imgsz=640,
    batch=4,
    name="first_dataset_retrain"
)

print("\nTraining complete!")
