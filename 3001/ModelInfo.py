
"""
YOLO Model Architecture and Information.

Loads each of the three trained YOLO models and displays
information about their architectures and model complexity.

Models:
    Model 1: Trained using the AL dataset.
    Model 2: Trained using the AL2 dataset.
    Model 3: Trained using the AL3 dataset.

Process:
    Loads each model from its saved weights file and calls
    the model.info() method to display model information.

Outputs:
    Prints information for each model, allowing comparison
    of their architectures and complexity.
"""

from ultralytics import YOLO

models = {
    "Model 1": "runs/detect/train-6/weights/best.pt",
    "Model 2": "runs/detect/runs/detect/apple_model_32/weights/best.pt",
    "Model 3": "runs/detect/runs/apple_model_64-9/weights/best.pt"
}

for name, path in models.items():
    print("\n", name)
    model = YOLO(path)
    model.info()
