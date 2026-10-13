
from ultralytics import YOLO

# Load pretrained YOLO11 nano model
model = YOLO("yolo11n.pt")

# Train using the AL3 dataset
model.train(
    data="AL3/apple3.yaml",
    epochs=100,
    imgsz=640,
    batch=4,
    project="runs",
    name="apple_model_64"
)

print("\nTraining complete!")

# Evaluate model
metrics = model.val(data="AL3/apple3.yaml")

print("\nValidation results:")
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)