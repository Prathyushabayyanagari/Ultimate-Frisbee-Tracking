
import cv2
from ultralytics import YOLO

VIDEO_PATH = "data/video/demo.mp4"
MODEL_PATH = "checkpoints/yolov10s.pt"

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video:", VIDEO_PATH)
    raise SystemExit(1)

print("Video opened successfully.")
print("Checking frisbee and sports-ball detections...")

for seconds in [1, 5, 10, 15, 20, 30]:
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    success, frame = cap.read()

    if not success:
        print(f"{seconds}s: Frame unavailable")
        continue

    result = model.predict(source=frame,imgsz=1280,conf=0.01,verbose=False)[0]
    objects = []

    for box in result.boxes:
        class_id = int(box.cls[0])

        if class_id in [29, 32]:
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            objects.append(
                (class_name, round(confidence, 3))
            )

    print(f"{seconds}s: {objects}")

cap.release()
print("Test completed.")