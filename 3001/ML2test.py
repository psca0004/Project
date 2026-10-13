
from ultralytics import YOLO

# Load new trained model
model = YOLO(
    "runs/detect/runs/detect/apple_model_32/weights/best.pt"
)

# Test on unseen images
results = model.predict(
    source="AL2/test/images",
    conf=0.25,
    save=True,
    project="runs/detect",
    name="apple_test_32"
)

print("\nDetection results:")

for result in results:
    print(
        result.path,
        "→",
        len(result.boxes),
        "apples detected"
    )