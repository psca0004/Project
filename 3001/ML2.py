
from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Train on the new AL2 dataset
model.train(
    data="AL2/apple.yaml",
    epochs=100,
    imgsz=640,
    batch=4,
    project="runs/detect",
    name="apple_model_32"
)

print("\nTraining complete!")