
"""
Machine Learning Model 3 - Training.

Trains a pretrained YOLO11n model to detect and locate apples
in RGB images using the AL3 dataset, which contains more
training images than the datasets used for Models 1 and 2.

Dataset:
    Training images: 56
    Validation images: 16
    Testing images: 8

Training Configuration:
    Model: YOLO11n
    Epochs: 100
    Image size: 640 pixels
    Batch size: 4

Process:
    Identifies the project directory using the location of
    the current Python script.

    Locates the AL3 dataset configuration file and uses it
    to train the pretrained YOLO11n model.

Outputs:
    Saves the trained model weights and training results
    under the project's 'runs/detect/apple_model_64' directory.

    Prints a message when training is complete.
"""

from ultralytics import YOLO
from pathlib import Path

# Find the project folder
project_folder = Path(__file__).resolve().parent

# Find the AL3 dataset configuration
dataset_yaml = project_folder / "AL3" / "apple3.yaml"

# Load the pretrained YOLO11 nano model
model = YOLO("yolo11n.pt")

# Train the model using the AL3 dataset
model.train(
    data=str(dataset_yaml),   # Dataset configuration
    epochs=100,               # Number of training cycles
    imgsz=640,                # Input image size
    batch=4,                  # Images per batch
    project=str(project_folder / "runs" / "detect"),
    name="apple_model_64"     # Results folder name
)

# Confirm training is complete
print("\nTraining complete!")
