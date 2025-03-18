from ultralytics import YOLO  ##type: ignore

model = YOLO("training/runs/detect/train/weights/best.pt")

result = model.predict("input_videos/08fd33_4.mp4", save=True)

print(result[0])

print("=" * 100)
for box in result[0].boxes:
    print(box)