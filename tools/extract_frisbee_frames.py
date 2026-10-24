"""Sample video frames into an unlabeled directory for frisbee annotation."""

import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract evenly spaced frames from game videos for manual "
            "frisbee labeling."
        )
    )
    parser.add_argument(
        "videos",
        nargs="+",
        help="Input video files to sample.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/frisbee_training/unlabeled",
        help="Directory for extracted JPG frames.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=1.0,
        help="Minimum time between saved frames.",
    )
    args = parser.parse_args()

    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be greater than zero.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for video_index, video_argument in enumerate(args.videos, start=1):
        video_path = Path(video_argument)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise OSError(f"Could not open video: {video_path}")

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if fps <= 0:
                raise ValueError(
                    f"Video reports invalid frame rate: {video_path}"
                )
            interval_frames = max(1, round(fps * args.interval_seconds))
            frame_index = 0
            next_frame = 0
            video_saved = 0

            while True:
                success, frame = capture.read()
                if not success:
                    break

                if frame_index >= next_frame:
                    output_path = output_dir / (
                        f"{video_index:02d}_{video_path.stem}_"
                        f"{frame_index:08d}.jpg"
                    )
                    if output_path.exists():
                        raise FileExistsError(
                            f"Refusing to overwrite existing frame: "
                            f"{output_path}"
                        )
                    if not cv2.imwrite(str(output_path), frame):
                        raise OSError(
                            f"Could not write extracted frame: {output_path}"
                        )
                    video_saved += 1
                    saved += 1
                    next_frame = frame_index + interval_frames

                frame_index += 1

            print(
                f"{video_path}: saved {video_saved} frame(s) "
                f"to {output_dir}"
            )
        finally:
            capture.release()

    print(f"Total frames saved: {saved}")


if __name__ == "__main__":
    main()
