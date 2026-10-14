# ============================================================
# MACHINE LEARNING MODEL 3 - TRAINING
#
# PURPOSE:
# Train a third YOLO model to detect apples in RGB images
# using a larger dataset than Models 1 and 2.
#
# DATASET:
# AL3 contains:
# - 56 training images
# - 16 validation images
# - 8 testing images
#
# TRAINING:
# - Model: YOLO11n
# - Epochs: 100
# - Image size: 640 pixels
# - Batch size: 4
# ============================================================

from ultralytics import YOLO

# Load the pretrained YOLO11 nano model
model = YOLO("yolo11n.pt")

# Train the model using the AL3 dataset
model.train(
    data="AL3/apple3.yaml",  # Dataset configuration
    epochs=100,              # Number of training cycles
    imgsz=640,               # Input image size
    batch=4,                 # Images per batch
    project="runs",          # Results directory
    name="apple_model_64"    # Results folder name
)

# Confirm training is complete
print("\nTraining complete!")
