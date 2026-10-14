# ============================================================
# MACHINE LEARNING MODEL 1 - TRAINING
#
# PURPOSE:
# Train a YOLO model to detect and locate apples in RGB images.
#
# DATASET:
# AL contains:
# - 11 training images
# - 3 validation images
# - 2 testing images
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

# Train the model using the AL dataset
model.train(
    data="AL/apple.yaml",       # Dataset configuration
    epochs=100,                 # Number of training cycles
    imgsz=640,                  # Input image size
    batch=4,                    # Images per batch
    name="first_dataset_retrain" # Results folder name
)

# Confirm training is complete
print("\nTraining complete!")