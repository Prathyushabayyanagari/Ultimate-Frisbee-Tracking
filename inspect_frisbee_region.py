
import os
import cv2

VIDEO_PATH = "data/video/demo.mp4"
OUTPUT_DIR = "outputs/manual_crops"

# Save one frame every 5 seconds.
TIMESTAMPS = [5, 10, 15, 20, 25, 30, 35, 40]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("ERROR: Could not open video:", VIDEO_PATH)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    print("Video opened successfully.")
    print("Video FPS:", fps)

    saved = 0

    for seconds in TIMESTAMPS:
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, frame = cap.read()

        if not ok:
            print("Could not read frame at", seconds, "seconds.")
            continue

        # Enlarge 2x for easier visual inspection.
        enlarged = cv2.resize(
            frame,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC,
        )

        filename = os.path.join(
            OUTPUT_DIR, f"frame_{seconds:02d}s.jpg"
        )

        if cv2.imwrite(filename, enlarged):
            saved += 1
            print("Saved:", filename)
        else:
            print("Failed to save:", filename)

    cap.release()
    print("Finished. Total frames saved:", saved)


if __name__ == "__main__":
    main()

import os
import cv2

VIDEO_PATH = "data/video/demo.mp4"
OUTPUT_DIR = "outputs/manual_crops"

# Save one frame every 5 seconds.
TIMESTAMPS = [5, 10, 15, 20, 25, 30, 35, 40]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("ERROR: Could not open video:", VIDEO_PATH)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    print("Video opened successfully.")
    print("Video FPS:", fps)

    saved = 0

    for seconds in TIMESTAMPS:
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, frame = cap.read()

        if not ok:
            print("Could not read frame at", seconds, "seconds.")
            continue

        # Enlarge 2x for easier visual inspection.
        enlarged = cv2.resize(
            frame,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC,
        )

        filename = os.path.join(
            OUTPUT_DIR, f"frame_{seconds:02d}s.jpg"
        )

        if cv2.imwrite(filename, enlarged):
            saved += 1
            print("Saved:", filename)
        else:
            print("Failed to save:", filename)

    cap.release()
    print("Finished. Total frames saved:", saved)


if __name__ == "__main__":
    main()
