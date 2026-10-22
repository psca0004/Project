
"""
Machine Learning Model 1 - Training.

Trains a pretrained YOLO11n model to detect and locate apples
in RGB images using the AL dataset.

Dataset:
    Training images: 11
    Validation images: 3
    Testing images: 2

Training Configuration:
    Model: YOLO11n
    Epochs: 100
    Image size: 640 pixels
    Batch size: 4

Outputs:
    Saves the trained model weights and training results
    to the 'first_dataset_retrain' experiment directory.
"""

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
