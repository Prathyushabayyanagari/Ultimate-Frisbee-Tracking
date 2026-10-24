
import cv2

video_path = "data/video/demo.mp4"
output_path = "outputs/frame_5s.jpg"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    raise SystemExit(1)

cap.set(cv2.CAP_PROP_POS_MSEC, 5000)
success, frame = cap.read()
cap.release()

if not success:
    print("ERROR: Could not read frame.")
    raise SystemExit(1)

cv2.imwrite(output_path, frame)
print("Saved frame to:", output_path)
print("Frame dimensions:", frame.shape)