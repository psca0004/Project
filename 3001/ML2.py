# ============================================================
# MACHINE LEARNING MODEL 2 - TRAINING
#
# PURPOSE:
# Train a second YOLO model to detect apples in RGB images
# using a larger dataset than Model 1.
#
# DATASET:
# AL2 contains:
# - 26 training images
# - 7 validation images
# - 4 testing images
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

# Train the model using the AL2 dataset
model.train(
    data="AL2/apple.yaml",     # Dataset configuration
    epochs=100,                # Number of training cycles
    imgsz=640,                 # Input image size
    batch=4,                   # Images per batch
    project="runs/detect",     # Results directory
    name="apple_model_32"      # Results folder name
)

# Confirm training is complete
print("\nTraining complete!")