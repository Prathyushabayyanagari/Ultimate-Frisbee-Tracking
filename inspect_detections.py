
import cv2
from ultralytics import YOLO

VIDEO_PATH = "data/video/demo.mp4"
MODEL_PATH = "checkpoints/yolov10s.pt"

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise SystemExit("Could not open video.")

for seconds in [5, 30]:
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    success, frame = cap.read()

    if not success:
        print(f"Could not read frame at {seconds}s")
        continue

    result = model.predict(
        frame,
        imgsz=1280,
        conf=0.01,
        verbose=False,
    )[0]

    # Draw only sports-ball and frisbee predictions.
    for box in result.boxes:
        class_id = int(box.cls[0])

        if class_id not in [29, 32]:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        name = model.names[class_id]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{name} {confidence:.3f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    output_path = f"outputs/inspection_{seconds}s.jpg"
    if cv2.imwrite(output_path, frame):
        print("Saved:", output_path)

cap.release()