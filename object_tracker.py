import time
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app, flags
from absl.flags import FLAGS

# from PIL import Image  # Not needed in batch mode
from tqdm import tqdm

# deep sort imports
from deep_sort import nn_matching, preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# YOLO model imports - support both YOLOv10 and YOLO26
from yolov10.config import YOLOv10Config
from yolov10.detection_adapter import DetectionAdapter as YOLOv10DetectionAdapter
from yolov10.inference import YOLOv10Inference
from yolov10.model_loader import ModelLoader as YOLOv10ModelLoader
from yolov26.config import YOLO26Config
from yolov26.detection_adapter import DetectionAdapter as YOLO26DetectionAdapter
from yolov26.inference import YOLO26Inference
from yolov26.model_loader import ModelLoader as YOLO26ModelLoader

# Removed core.utils dependency for YOLOv10 migration
# from core.config import cfg  # cfg not used in YOLOv10 version

# from tools import generate_detections as gdet  # TensorFlow dependency - using mock for testing

# Suppress ultralytics torch.load FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="ultralytics.*")

flags.DEFINE_string("framework", "yolov10", "(yolov10, yolo26, tf, tflite, trt")
flags.DEFINE_string("weights", "./checkpoints/yolov10s.pt", "path to weights file")
flags.DEFINE_integer("size", 640, "resize images to")
flags.DEFINE_boolean("tiny", False, "yolo or yolo-tiny (deprecated for yolov10)")
flags.DEFINE_string(
    "model",
    "yolov10s",
    "yolov10n, yolov10s, yolov10m, yolov10b, yolov10l, yolov10x, yolo26n, yolo26s, yolo26m, yolo26l, yolo26x",
)
flags.DEFINE_string(
    "video", "./data/video/test.mp4", "path to input video or set to 0 for webcam"
)
flags.DEFINE_string("output", None, "path to output video")
flags.DEFINE_string("output_df", None, "path prefix for CSV player locations")
flags.DEFINE_string(
    "output_format", "XVID", "codec used in VideoWriter when saving video to file"
)
flags.DEFINE_float("iou", 0.45, "iou threshold")
flags.DEFINE_float("score", 0.50, "score threshold")
flags.DEFINE_boolean("dont_show", False, "dont show video output")
flags.DEFINE_boolean("info", False, "show detailed info of tracked objects")
flags.DEFINE_boolean("count", False, "count objects being tracked on screen")
flags.DEFINE_boolean(
    "limits",
    False,
    "Limit to detections inside the field. TODO: Make interactive to define bounds.",
)
flags.DEFINE_string(
    "device", "auto", "device to run inference on (auto, cpu, cuda, mps)"
)
flags.DEFINE_boolean(
    "batch_mode", False, "Enable batch processing for fast video processing"
)
flags.DEFINE_integer("batch_size", 8, "Batch size for inference (default: 8)")
flags.DEFINE_boolean("preload_frames", True, "Preload all frames for optimal batching")
flags.DEFINE_integer("max_memory_mb", 2048, "Maximum memory for frame preloading in MB")

RECT_COORD = [[683, 519], [1259, 169], [844, 120], [110, 250]]
TARGET_COORD = [[0, 0], [280, 0], [280, 150], [0, 150]]
TRANSFORM_MAT = cv2.getPerspectiveTransform(
    np.float32(np.array(RECT_COORD)), np.float32(np.array(TARGET_COORD))
)


def draw_raw_disc_candidates(frame_rgb, detections):
    """Draw only unconfirmed model predictions named frisbee in orange."""
    height, width = frame_rgb.shape[:2]

    for detection in detections:
        class_name = str(detection["class_name"]).strip()
        if class_name.lower() != "frisbee":
            continue

        x1, y1, x2, y2 = map(int, detection["bbox"])
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        confidence = float(detection["confidence"])
        color_rgb = (255, 165, 0)
        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color_rgb, 2)
        cv2.putText(
            frame_rgb,
            f"{class_name} candidate {confidence:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_rgb,
            2,
        )

    return frame_rgb


def filter_player_detections(bboxes, scores, classes, class_mapping):
    """Keep disc candidates out of DeepSORT's player association step."""
    player_indices = np.asarray(
        [
            index
            for index, class_id in enumerate(classes)
            if class_mapping.get(int(class_id), "").strip().lower()
            == "person"
        ],
        dtype=np.intp,
    )
    return bboxes[player_indices], scores[player_indices], classes[player_indices]


def video_frames(capture, writer, progress):
    """Yield frames and release video resources on EOF, break, or error."""
    try:
        while True:
            success, frame = capture.read()
            if not success:
                yield None
                return
            yield frame
    finally:
        progress.close()
        if writer is not None:
            writer.release()
        capture.release()
        cv2.destroyAllWindows()


def estimate_video_memory(video_path: str, max_memory_mb: int) -> bool:
    """
    Estimate if video can be preloaded into memory.

    Args:
        video_path: Path to video file
        max_memory_mb: Maximum memory in MB

    Returns:
        True if video can be preloaded
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()

        # Estimate memory usage (assuming 3 channels, uint8)
        frame_size_mb = (width * height * 3) / (1024 * 1024)
        total_memory_mb = frame_size_mb * total_frames

        print(f"Video: {width}x{height}, {total_frames} frames")
        print(f"Estimated memory needed: {total_memory_mb:.1f} MB")
        print(f"Memory limit: {max_memory_mb} MB")

        return total_memory_mb <= max_memory_mb

    except Exception as e:
        print(f"Error estimating video memory: {e}")
        return False


def should_preload_frames(
    video_path: str, max_memory_mb: int, preload_override: bool = False
) -> bool:
    """
    Decide whether to preload frames based on video size and memory limit.

    Args:
        video_path: Path to video file
        max_memory_mb: Maximum memory in MB
        preload_override: Override flag from user

    Returns:
        True if frames should be preloaded
    """
    if preload_override is not None:
        return preload_override

    return estimate_video_memory(video_path, max_memory_mb)


def run_batch_inference(
    frames: list, yolo_inference, detection_adapter, batch_size: int = 4
) -> tuple:
    """
    Run YOLO inference on a batch of frames with sub-batching for memory safety.

    Args:
        frames: List of frames to process
        yolo_inference: YOLO inference object (YOLOv10 or YOLO26)
        detection_adapter: Detection adapter for DeepSORT compatibility
        batch_size: Sub-batch size for memory management

    Returns:
        Tuple of (formatted_detections, timings)
    """
    import time

    timings = {"inference": [], "formatting": []}
    formatted_detections = []

    # Process frames in sub-batches for memory safety
    for i in range(0, len(frames), batch_size):
        sub_batch = frames[i : i + batch_size]

        # Run batch inference
        start_time = time.time()
        batch_results = yolo_inference.infer_batch(sub_batch)
        inference_time = time.time() - start_time
        timings["inference"].append(inference_time)

        # Format detections for DeepSort
        format_start = time.time()
        for j, (frame, results) in enumerate(zip(sub_batch, batch_results)):
            bboxes, scores, classes, num_objects = (
                detection_adapter.format_for_deepsort(results, frame.shape)
            )
            formatted_detections.append(
                {
                    "bboxes": bboxes,
                    "scores": scores,
                    "classes": classes,
                    "num_objects": num_objects,
                    "raw_detections": results,
                    "frame_idx": i + j,
                }
            )
        format_time = time.time() - format_start
        timings["formatting"].append(format_time)

    return formatted_detections, timings


def process_video_batch(
    video_path: str,
    output_path: str,
    yolo_inference,
    detection_adapter,
    tracker,
    encoder,
    batch_size: int = 8,
    preload_frames: bool = True,
    max_memory_mb: int = 2048,
    total_frames: int = 0,
):
    """
    Process entire video with batch YOLO inference and sequential DeepSort processing.

    Args:
        video_path: Path to input video
        output_path: Path to output video (optional)
        yolo_inference: YOLO inference object
        detection_adapter: Detection adapter for DeepSORT
        tracker: DeepSort tracker object
        encoder: Feature encoder for DeepSort
        batch_size: Main batch size for processing
        preload_frames: Whether to preload all frames
        max_memory_mb: Memory limit for frame preloading
        total_frames: Total frame count (for progress tracking)
    """
    import time

    # Initialize video capture
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    if total_frames is None:
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup output video if needed
    out = None
    if output_path:
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    # Performance tracking
    timings = {
        "batch_inference": [],
        "deepsort_processing": [],
        "drawing": [],
        "total_frame": [],
    }

    # frame_num = 0  # Not needed in batch mode
    pbar = tqdm(total=total_frames, desc="Batch processing frames", unit="frames")

    # Strategy: Preload or stream
    all_frames = []  # Initialize outside conditional
    if preload_frames:
        print("Reading all video frames into memory...")
        all_frames = []
        while True:
            ret, frame = vid.read()
            if not ret:
                break
            all_frames.append(frame)
        vid.release()
        max_frames = len(all_frames)
        print(f"Loaded {max_frames} frames into memory")
    else:
        print("Streaming frames in batches...")
        max_frames = total_frames

    try:
        # Process frames in batches
        for batch_start in range(0, max_frames, batch_size):
            batch_end = min(batch_start + batch_size, max_frames)

            # Get frames for this batch
            if preload_frames:
                batch_frames = all_frames[batch_start:batch_end]
            else:
                # Stream frames for this batch
                batch_frames = []
                vid = cv2.VideoCapture(video_path)
                vid.set(cv2.CAP_PROP_POS_FRAMES, batch_start)
                for _ in range(batch_end - batch_start):
                    ret, frame = vid.read()
                    if not ret:
                        break
                    batch_frames.append(frame)
                vid.release()

            # Batch YOLO inference
            batch_start_time = time.time()
            formatted_detections, batch_timings = run_batch_inference(
                batch_frames, yolo_inference, detection_adapter
            )
            batch_time = time.time() - batch_start_time
            timings["batch_inference"].append(batch_time)

            # Sequential DeepSort processing (maintains tracking consistency)
            for i, (frame, detections) in enumerate(
                zip(batch_frames, formatted_detections)
            ):
                # frame_num = batch_start + i  # Not needed in batch mode
                frame_start_time = time.time()

                # Extract detection data
                bboxes = detections["bboxes"]
                scores = detections["scores"]
                classes = detections["classes"]
                num_objects = detections["num_objects"]

                class_mapping = {
                    int(detection["class_id"]): str(detection["class_name"])
                    for detection in detections["raw_detections"]
                }
                bboxes, scores, classes = filter_player_detections(
                    bboxes, scores, classes, class_mapping
                )
                num_objects = len(classes)

                names = []
                deleted_indx = []

                # Apply field limits if enabled
                if FLAGS.limits:
                    for i in range(num_objects):
                        y_max_transformed = cv2.perspectiveTransform(
                            np.array(
                                [[bboxes[i][0], bboxes[i][1] + bboxes[i][3]]],
                                dtype="float32",
                            )[None, :, :],
                            TRANSFORM_MAT,
                        ).flatten()[-1]
                        if y_max_transformed > 150:
                            deleted_indx.append(i)

                for i in range(num_objects):
                    if i not in deleted_indx:
                        class_indx = int(classes[i])
                        class_name = class_mapping.get(
                            class_indx, f"class_{class_indx}"
                        )
                        names.append(class_name)

                names = np.array(names) if names else np.array([])

                # Handle object count display
                count = len(names)
                if FLAGS.count:
                    print(f"Objects being tracked: {count}")

                # Delete detections that are outside limits
                if deleted_indx:
                    bboxes = np.delete(bboxes, deleted_indx, axis=0)
                    scores = np.delete(scores, deleted_indx, axis=0)
                    names = np.delete(names, deleted_indx, axis=0)

                # Feature extraction and DeepSort update
                features = encoder(frame, bboxes)
                detections_list = [
                    Detection(bbox, score, class_name, feature)
                    for bbox, score, class_name, feature in zip(
                        bboxes, scores, names, features
                    )
                ]

                # DeepSort processing
                deepsort_start = time.time()
                tracker.predict()
                tracker.update(detections_list)
                timings["deepsort_processing"].append(time.time() - deepsort_start)

                # Drawing for output video
                draw_start = time.time()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                draw_raw_disc_candidates(
                    frame_rgb, detections["raw_detections"]
                )

                # Initialize color map
                cmap = plt.get_cmap("tab20b")
                colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()
                    class_name = track.get_class()

                    color = colors[int(track.track_id) % len(colors)]
                    color = [i * 255 for i in color]
                    cv2.rectangle(
                        frame_rgb,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])),
                        color,
                        2,
                    )

                    # Add label with track ID
                    label = f"{class_name[0] + '-' + str(track.track_id) if len(class_name) > 0 else f'track-{track.track_id}'}"
                    cv2.rectangle(
                        frame_rgb,
                        (int(bbox[0]), int(bbox[1] - 30)),
                        (
                            int(bbox[0])
                            + (len(class_name) + len(str(track.track_id))) * 17,
                            int(bbox[1]),
                        ),
                        color,
                        -1,
                    )
                    cv2.putText(
                        frame_rgb,
                        label,
                        (int(bbox[0]), int(bbox[1] - 10)),
                        0,
                        0.75,
                        (255, 255, 255),
                        2,
                    )

                    if FLAGS.info:
                        print(
                            f"Tracker ID: {str(track.track_id)}, Class: {class_name},  BBox Coords (xmin, ymin, xmax, ymax): {(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))}"
                        )

                timings["drawing"].append(time.time() - draw_start)
                timings["total_frame"].append(time.time() - frame_start_time)

                # Save frame if output specified
                if output_path:
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)

                pbar.update(1)

    finally:
        # Cleanup
        if out:
            out.release()
        if vid.isOpened():
            vid.release()
        pbar.close()

    # Print performance summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING PERFORMANCE SUMMARY")
    print("=" * 80)

    for operation, times in timings.items():
        if times:
            avg_time = np.mean(times) * 1000  # Convert to ms
            total_time = sum(times)
            print(f"{operation:20}: {avg_time:6.2f}ms avg, {total_time:.2f}s total")

    # Calculate effective FPS
    if timings["total_frame"]:
        total_processing_time = sum(timings["total_frame"])
        effective_fps = total_frames / total_processing_time
        print(f"\n{'Effective FPS':20}: {effective_fps:.1f}")
        print(f"{'Total frames':20}: {total_frames}")
        print(f"{'Batch size':20}: {batch_size}")

        # Compare with estimated sequential processing
        avg_frame_time = np.mean(timings["total_frame"]) * 1000
        estimated_sequential_fps = 1000 / avg_frame_time if avg_frame_time > 0 else 0
        speedup = (
            effective_fps / estimated_sequential_fps
            if estimated_sequential_fps > 0
            else 0
        )
        print(f"{'Speedup vs sequential':20}: {speedup:.2f}x")


def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    # initialize deep sort
    # For YOLOv10 testing, use a simple mock encoder
    # In production, this should be replaced with a PyTorch-based encoder
    def mock_encoder(image, boxes):
        """Mock encoder that generates random features for testing."""
        import numpy as np

        return np.random.rand(len(boxes), 128).astype(np.float32)

    encoder = mock_encoder
    # model_filename = "model_data/mars-small128.pb"
    # encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget
    )
    # initialize tracker
    tracker = Tracker(metric, max_age=60)

    # Validate local inputs before loading model weights or opening video.
    video_path = FLAGS.video
    is_camera = video_path.isdigit()
    if not is_camera and not Path(video_path).is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if not Path(FLAGS.weights).is_file():
        raise FileNotFoundError(f"Model weights not found: {FLAGS.weights}")

    # Determine which YOLO model to use based on model flag
    model_variant = FLAGS.model.lower()

    if model_variant.startswith("yolo26"):
        # Initialize YOLO26
        yolo_config = YOLO26Config.from_args(FLAGS)
        yolo_config.validate()

        # Load YOLO26 model
        model_loader = YOLO26ModelLoader(
            model_variant=yolo_config.model_variant,
            device=yolo_config.device,
            confidence_threshold=yolo_config.confidence_threshold,
            iou_threshold=yolo_config.iou_threshold,
            nms_free=yolo_config.nms_free,
        )

        model = model_loader.load_model(FLAGS.weights)
        model_loader.optimize_for_inference()

        # Initialize inference and detection adapter
        yolo_inference = YOLO26Inference(model, yolo_config)
        detection_adapter = YOLO26DetectionAdapter(
            target_classes=["person", "sports ball"],
            confidence_threshold=yolo_config.confidence_threshold,
            nms_free=yolo_config.nms_free,
        )

        print(
            f"YOLO26 {yolo_config.model_variant} loaded with NMS-free inference: {yolo_config.nms_free}"
        )

    else:
        # Initialize YOLOv10 (default)
        yolo_config = YOLOv10Config.from_args(FLAGS)
        yolo_config.validate()

        # Load YOLOv10 model
        model_loader = YOLOv10ModelLoader(
            model_variant=yolo_config.model_variant,
            device=yolo_config.device,
            confidence_threshold=yolo_config.confidence_threshold,
            iou_threshold=yolo_config.iou_threshold,
        )

        model = model_loader.load_model(FLAGS.weights)
        model_loader.optimize_for_inference()

        # Initialize inference and detection adapter
        yolo_inference = YOLOv10Inference(model, yolo_config)
        detection_adapter = YOLOv10DetectionAdapter(
            target_classes=["person", "frisbee", "sports ball"],
            confidence_threshold=yolo_config.confidence_threshold,
        )

        print(f"YOLOv10 {yolo_config.model_variant} loaded")

    # begin video capture
    video_source = int(video_path) if is_camera else video_path
    vid = cv2.VideoCapture(video_source)
    if not vid.isOpened():
        vid.release()
        raise OSError(f"Could not open video source: {video_path}")

    out = None
    csv_rows = []

    # get video ready to save locally if flag is set
    if FLAGS.output:
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(vid.get(cv2.CAP_PROP_FPS))
        if width <= 0 or height <= 0:
            vid.release()
            raise ValueError("Video source reported invalid frame dimensions.")
        if fps <= 0:
            fps = 30.0
        output_path = Path(FLAGS.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if len(FLAGS.output_format) != 4:
            vid.release()
            raise ValueError("Output codec must be a four-character code.")
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(
            str(output_path), codec, fps, (width, height)
        )
        if not out.isOpened():
            out.release()
            vid.release()
            raise RuntimeError(
                f"Could not open output video writer for {output_path} "
                f"with codec {FLAGS.output_format!r}."
            )
    if FLAGS.output_df:
        Path(FLAGS.output_df).parent.mkdir(parents=True, exist_ok=True)

    # Get total frame count for progress bar
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # BATCH MODE INTEGRATION
    if FLAGS.batch_mode:
        if out is not None:
            out.release()
            out = None
        vid.release()
        print("\n" + "=" * 60)
        print("BATCH PROCESSING MODE ENABLED")
        print("=" * 60)
        print("Video display suppressed for maximum performance")
        print(f"Batch size: {FLAGS.batch_size}")
        print(f"Memory limit: {FLAGS.max_memory_mb} MB")
        print("=" * 60 + "\n")

        # Auto-hide display for maximum performance
        original_dont_show = FLAGS.dont_show
        FLAGS.dont_show = True

        # Decide memory strategy
        should_preload = should_preload_frames(
            video_path, FLAGS.max_memory_mb, FLAGS.preload_frames
        )

        if should_preload:
            print("Strategy: Preloading all frames for maximum performance")
        else:
            print("Strategy: Streaming batches to handle large video")

        # Update configs with batch size
        if model_variant.startswith("yolo26"):
            yolo_config.batch_size = FLAGS.batch_size
        else:
            yolo_config.batch_size = FLAGS.batch_size

        # Process video in batch mode and exit
        try:
            process_video_batch(
                video_path=video_path,
                output_path=FLAGS.output,
                yolo_inference=yolo_inference,
                detection_adapter=detection_adapter,
                tracker=tracker,
                encoder=encoder,
                batch_size=FLAGS.batch_size,
                preload_frames=should_preload,
                max_memory_mb=FLAGS.max_memory_mb,
                total_frames=total_frames,
            )
        finally:
            FLAGS.dont_show = original_dont_show
            cv2.destroyAllWindows()
        return  # Exit main function after batch processing

    frame_num = 0
    # Initialize progress bar
    pbar = tqdm(total=total_frames, desc="Processing frames", unit="frames")
    # while video is running
    for frame_bgr in video_frames(vid, out, pbar):
        if frame_bgr is None:
            print("Video ended or an unreadable frame was encountered.")
            break
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_num += 1
        start_time = time.time()

        # run YOLOv10 detections
        yolo_detections = yolo_inference.infer_single(frame_bgr)

        # convert YOLOv10 detections to DeepSORT format
        bboxes, scores, classes, num_objects = detection_adapter.format_for_deepsort(
            yolo_detections, frame.shape
        )

        class_mapping = {
            int(detection["class_id"]): str(detection["class_name"])
            for detection in yolo_detections
        }
        bboxes, scores, classes = filter_player_detections(
            bboxes, scores, classes, class_mapping
        )
        num_objects = len(classes)

        # DeepSORT receives players only; disc candidates remain independent.
        names = []
        deleted_indx = []

        if FLAGS.limits:
            for i in range(num_objects):
                y_max_transformed = cv2.perspectiveTransform(
                    np.array(
                        [[bboxes[i][0], bboxes[i][1] + bboxes[i][3]]],
                        dtype="float32",
                    )[None, :, :],
                    TRANSFORM_MAT,
                ).flatten()[-1]
                if y_max_transformed > 150:
                    deleted_indx.append(i)

        for i in range(num_objects):
            if i not in deleted_indx:
                class_indx = int(classes[i])
                class_name = class_mapping.get(class_indx, f"class_{class_indx}")
                names.append(class_name)

        names = np.array(names) if names else np.array([])
        count = len(names)
        if FLAGS.count:
            cv2.putText(
                frame,
                f"Objects being tracked: {count}",
                (5, 35),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                2,
                (0, 255, 0),
                2,
            )
            print(f"Objects being tracked: {count}")

        # delete detections that are outside limits
        if deleted_indx:
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)
            names = np.delete(names, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [
            Detection(bbox, score, class_name, feature)
            for bbox, score, class_name, feature in zip(bboxes, scores, names, features)
        ]

        # initialize color map
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(
            boxs, classes, nms_max_overlap, scores
        )
        detections = [detections[i] for i in indices]

        draw_raw_disc_candidates(frame, yolo_detections)

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()

            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(
                frame,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color,
                2,
            )
            cv2.rectangle(
                frame,
                (int(bbox[0]), int(bbox[1] - 30)),
                (
                    int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17,
                    int(bbox[1]),
                ),
                color,
                -1,
            )
            cv2.putText(
                frame,
                class_name[0] + "-" + str(track.track_id),
                (int(bbox[0]), int(bbox[1] - 10)),
                0,
                0.75,
                (255, 255, 255),
                2,
            )

            # if enable info flag then print details about each track
            if FLAGS.info:
                print(
                    f"Tracker ID: {str(track.track_id)}, Class: {class_name},  BBox Coords (xmin, ymin, xmax, ymax): {(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))}"
                )

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        pbar.set_description(f"Processing frames - FPS: {round(fps, 1)}")
        pbar.update(1)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)

        # if output_df flag is set, save player locations as csv.
        if FLAGS.output_df:
            csv_rows.append(
                {
                    **{
                        track.get_class() + "-" + str(track.track_id) + "x": int(
                            0.5 * (track.to_tlbr()[0] + track.to_tlbr()[2])
                        )
                        for track in tracker.tracks
                        if track.is_confirmed() and track.time_since_update <= 1
                    },
                    **{
                        track.get_class()
                        + "-"
                        + str(track.track_id)
                        + "y": track.to_tlbr()[3]
                        for track in tracker.tracks
                        if track.is_confirmed() and track.time_since_update <= 1
                    },
                }
            )

        if not FLAGS.dont_show and cv2.waitKey(1) & 0xFF == ord("q"):
            break
    if FLAGS.output_df:
        pd.DataFrame.from_records(csv_rows).to_csv(
            f"{FLAGS.output_df}" + ("-tiny" if FLAGS.tiny else "") + ".csv"
        )


if __name__ == "__main__":
    app.run(main)
