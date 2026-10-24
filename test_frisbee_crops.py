"""Compare full-frame and overlapping-tile COCO frisbee detections."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from deep_sort.preprocessing import non_max_suppression


TARGET_CLASSES = [29, 32]


def make_crop_positions(length, crop_size, overlap):
    """Return crop starting positions, including the far edge."""
    if length <= crop_size:
        return [0]

    step = max(1, crop_size - overlap)
    positions = list(range(0, length - crop_size + 1, step))
    final_position = length - crop_size
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def class_name(names, class_id):
    if isinstance(names, dict):
        return names.get(class_id, f"class_{class_id}")
    if class_id < len(names):
        return names[class_id]
    return f"class_{class_id}"


def detect_candidates(model, image, offset, source, image_size, confidence):
    results = model.predict(
        source=image,
        imgsz=image_size,
        conf=confidence,
        classes=TARGET_CLASSES,
        device="cpu",
        verbose=False,
    )
    candidates = []
    if not results or results[0].boxes is None:
        return candidates

    offset_x, offset_y = offset
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        class_id = int(box.cls[0].cpu())
        candidates.append(
            {
                "bbox": [
                    x1 + offset_x,
                    y1 + offset_y,
                    x2 + offset_x,
                    y2 + offset_y,
                ],
                "confidence": float(box.conf[0].cpu()),
                "class_id": class_id,
                "class_name": class_name(model.names, class_id),
                "source": source,
            }
        )
    return candidates


def deduplicate_candidates(candidates, frame_shape, overlap_threshold=0.45):
    """Suppress duplicate boxes from overlapping tiles without mixing classes."""
    if not candidates:
        return []

    height, width = frame_shape[:2]
    boxes = []
    valid_candidates = []
    for candidate in candidates:
        x1, y1, x2, y2 = candidate["bbox"]
        x1 = max(0.0, min(width - 1.0, x1))
        y1 = max(0.0, min(height - 1.0, y1))
        x2 = max(0.0, min(width - 1.0, x2))
        y2 = max(0.0, min(height - 1.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        candidate["bbox"] = [x1, y1, x2, y2]
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        valid_candidates.append(candidate)

    if not valid_candidates:
        return []
    box_array = np.asarray(boxes, dtype=np.float64)
    classes = np.asarray([item["class_id"] for item in valid_candidates])
    scores = np.asarray([item["confidence"] for item in valid_candidates])
    kept = non_max_suppression(
        box_array, classes, overlap_threshold, scores
    )
    return [valid_candidates[index] for index in kept]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare full-frame COCO frisbee/sports-ball predictions with "
            "predictions from overlapping high-resolution crops."
        )
    )
    parser.add_argument("--video", default="data/video/demo.mp4")
    parser.add_argument("--weights", default="checkpoints/yolov10s.pt")
    parser.add_argument("--output-dir", default="outputs/frisbee_diagnostics")
    parser.add_argument(
        "--timestamps", nargs="+", type=float, default=[5, 15, 30]
    )
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tile-imgsz", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.01)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing diagnostic images in the output directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video)
    weights_path = Path(args.weights)
    output_dir = Path(args.output_dir)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    if args.tile_size <= 0 or not 0.0 <= args.overlap < 1.0:
        raise ValueError("tile-size must be positive and overlap in [0, 1).")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading local model: {weights_path}")
    model = YOLO(str(weights_path))
    print(
        "Model class mapping:",
        {
            class_id: class_name(model.names, class_id)
            for class_id in (0, 29, 32)
        },
    )

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        video.release()
        raise OSError(f"Could not open video: {video_path}")

    fps = float(video.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        video.release()
        raise ValueError(f"Video reports invalid frame rate: {fps}")

    try:
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Video reports invalid frame dimensions.")
        crop_size = min(args.tile_size, frame_width, frame_height)
        overlap = min(crop_size - 1, int(crop_size * args.overlap))
        x_positions = make_crop_positions(
            frame_width, crop_size, overlap
        )
        y_positions = make_crop_positions(
            frame_height, crop_size, overlap
        )
        print(
            f"Video: {frame_width}x{frame_height} at {fps:.2f} FPS; "
            f"{len(x_positions) * len(y_positions)} tiles per sampled frame."
        )

        for seconds in args.timestamps:
            video.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            success, frame = video.read()
            if not success:
                print(f"{seconds:g}s: could not read frame")
                continue

            full_frame = detect_candidates(
                model,
                frame,
                offset=(0, 0),
                source="full",
                image_size=args.imgsz,
                confidence=args.confidence,
            )
            tiled = []
            for y in y_positions:
                for x in x_positions:
                    crop = frame[
                        y : min(y + crop_size, frame_height),
                        x : min(x + crop_size, frame_width),
                    ]
                    tiled.extend(
                        detect_candidates(
                            model,
                            crop,
                            offset=(x, y),
                            source=f"tile({x},{y})",
                            image_size=args.tile_imgsz,
                            confidence=args.confidence,
                        )
                    )

            merged = deduplicate_candidates(
                full_frame + tiled, frame.shape
            )
            print(
                f"{seconds:g}s: {len(full_frame)} full-frame, "
                f"{len(tiled)} tile, {len(merged)} unique candidate(s)"
            )
            frisbee_candidates = [
                candidate
                for candidate in merged
                if str(candidate["class_name"]).strip().lower() == "frisbee"
            ]
            for candidate in merged:
                x1, y1, x2, y2 = candidate["bbox"]
                print(
                    f"  {candidate['class_name']} "
                    f"confidence={candidate['confidence']:.3f} "
                    f"box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) "
                    f"source={candidate['source']}"
                )

            for candidate in frisbee_candidates:
                x1, y1, x2, y2 = candidate["bbox"]
                orange_bgr = (0, 165, 255)
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    orange_bgr,
                    2,
                )
                cv2.putText(
                    frame,
                    f"{candidate['class_name']} {candidate['confidence']:.2f}",
                    (int(x1), max(20, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    orange_bgr,
                    2,
                )

            if not frisbee_candidates:
                cv2.putText(
                    frame,
                    "No frisbee-class detections",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            output_path = output_dir / f"crop_test_{seconds:g}s.jpg"
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite diagnostic image: {output_path}; "
                    "choose another --output-dir or pass --overwrite."
                )
            if not cv2.imwrite(str(output_path), frame):
                raise OSError(f"Could not save diagnostic image: {output_path}")
            print(f"  saved {output_path}")
    finally:
        video.release()


if __name__ == "__main__":
    main()
