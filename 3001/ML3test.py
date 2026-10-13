
from ultralytics import YOLO

# Load the third trained model
model = YOLO(
    "runs/detect/runs/apple_model_64-9/weights/best.pt"
)

# Test on unseen images
results = model.predict(
    source="AL3/test/images",
    conf=0.25,
    save=True,
    project="runs/detect",
    name="apple_test_64"
)

print("\nDetection results:")

total_apples = 0

for result in results:

    apple_count = len(result.boxes)
    total_apples += apple_count

    print(
        result.path,
        "→",
        apple_count,
        "apples detected"
    )

print("\nSummary:")
print("Images tested:", len(results))
print("Total apples detected:", total_apples)
print("Average apples per image:", total_apples / len(results))