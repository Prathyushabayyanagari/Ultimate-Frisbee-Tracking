"""Prepare or train a COCO-class-compatible frisbee detector."""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def coco_names(model):
    names = model.names
    if isinstance(names, dict):
        return {int(class_id): str(name) for class_id, name in names.items()}
    return {index: str(name) for index, name in enumerate(names)}


def write_dataset_yaml(dataset_root, names):
    yaml_path = dataset_root / "data.yaml"
    if yaml_path.exists():
        return yaml_path

    lines = [
        f"path: {json.dumps(dataset_root.resolve().as_posix())}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines.extend(
        f"  {class_id}: {json.dumps(name)}"
        for class_id, name in sorted(names.items())
    )
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def validate_labels(dataset_root):
    for split in ("train", "val"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        images = [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if not images:
            raise ValueError(f"No images found in {image_dir}")

        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(
                    f"Missing YOLO label file for {image_path}: {label_path}"
                )
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 5:
                    raise ValueError(
                        f"{label_path}:{line_number}: expected "
                        "class x_center y_center width height"
                    )
                class_id = int(fields[0])
                values = [float(value) for value in fields[1:]]
                if class_id not in (0, 29):
                    raise ValueError(
                        f"{label_path}:{line_number}: use class 0 for "
                        "person or class 29 for frisbee"
                    )
                if any(value < 0 or value > 1 for value in values):
                    raise ValueError(
                        f"{label_path}:{line_number}: box coordinates "
                        "must be normalized to [0, 1]"
                    )
                if values[2] <= 0 or values[3] <= 0:
                    raise ValueError(
                        f"{label_path}:{line_number}: box width and height "
                        "must be positive"
                    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a COCO-aligned dataset config or fine-tune the local "
            "YOLOv10s checkpoint after labeling video frames."
        )
    )
    parser.add_argument(
        "--dataset",
        default="data/frisbee_training",
        help="Dataset root containing images/ and labels/.",
    )
    parser.add_argument(
        "--weights",
        default="checkpoints/yolov10s.pt",
        help="Existing local YOLOv10 checkpoint to fine-tune.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    weights_path = Path(args.weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch <= 0:
        parser.error("--epochs, --imgsz, and --batch must be positive.")

    print(f"Loading local checkpoint: {weights_path}")
    model = YOLO(str(weights_path))
    names = coco_names(model)
    if (
        names.get(0) != "person"
        or names.get(29) != "frisbee"
        or names.get(32) != "sports ball"
    ):
        raise ValueError(
            "The selected checkpoint does not use the expected COCO "
            "mapping (0=person, 29=frisbee, 32=sports ball)."
        )

    for split in ("train", "val"):
        (dataset_root / "images" / split).mkdir(
            parents=True, exist_ok=True
        )
        (dataset_root / "labels" / split).mkdir(
            parents=True, exist_ok=True
        )

    yaml_path = write_dataset_yaml(dataset_root, names)
    print(f"Dataset config: {yaml_path}")
    if args.prepare_only:
        print(
            "Layout ready. Add images and YOLO label files before training; "
            "class IDs must be 0=person and 29=frisbee."
        )
        return

    validate_labels(dataset_root)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="outputs/frisbee_training",
        name="yolov10s-frisbee",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
