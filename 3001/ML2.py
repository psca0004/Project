
"""
Machine Learning Model 2 - Training.

Trains a pretrained YOLO11n model to detect and locate apples
in RGB images using the larger AL2 dataset.

Dataset:
    Training images: 26
    Validation images: 7
    Testing images: 4

Training Configuration:
    Model: YOLO11n
    Epochs: 100
    Image size: 640 pixels
    Batch size: 4

Outputs:
    Saves the trained model weights and training results
    under the 'runs/detect/apple_model_32' directory.

    Prints a message when training is complete.
"""

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
