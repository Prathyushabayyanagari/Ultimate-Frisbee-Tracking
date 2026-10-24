"""
YOLO26 Utility Functions

Helper functions for YOLO26 integration and common operations.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def letterbox_image(
    image: np.ndarray,
    new_shape: Union[int, Tuple[int, int]] = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize and pad image while maintaining aspect ratio.

    Args:
        image: Input image (BGR format)
        new_shape: Target size (int for square, or (width, height))
        color: Padding color (BGR)

    Returns:
        Tuple of (letterboxed_image, scale_ratio, (pad_width, pad_height))
    """
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Original shape
    shape = image.shape[:2]  # height, width
    h, w = shape

    # Target shape
    new_h, new_w = new_shape

    # Scale ratio (new / old)
    scale = min(new_h / h, new_w / w)

    # Compute padding
    new_unpad = int(round(w * scale)), int(round(h * scale))
    pad_w, pad_h = new_w - new_unpad[0], new_h - new_unpad[1]

    # Divide padding into two sides
    pad_w /= 2
    pad_h /= 2

    # Resize and pad
    if shape[::-1] != new_unpad:  # resize if needed
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))

    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return image, scale, (pad_w, pad_h)


def scale_boxes(
    boxes: np.ndarray, image_shape: Tuple[int, int], input_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Scale boxes from input image space to original image space.

    Args:
        boxes: Bounding boxes in input space [x1, y1, x2, y2]
        image_shape: Original image shape (height, width)
        input_shape: Model input shape (height, width)

    Returns:
        Scaled boxes in original image space
    """
    if len(boxes) == 0:
        return boxes

    # Calculate gain
    gain = min(input_shape[0] / image_shape[0], input_shape[1] / image_shape[1])

    # Calculate padding
    pad = (
        (input_shape[1] - image_shape[1] * gain) / 2,
        (input_shape[0] - image_shape[0] * gain) / 2,
    )

    # Scale boxes
    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain

    # Clip boxes
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image_shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image_shape[0])  # y1, y2

    return boxes


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from xyxy to xywh format.

    Args:
        boxes: Bounding boxes in [x1, y1, x2, y2] format

    Returns:
        Bounding boxes in [x, y, w, h] format
    """
    if len(boxes) == 0:
        return boxes

    converted = boxes.copy()
    converted[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
    converted[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
    return converted


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from xywh to xyxy format.

    Args:
        boxes: Bounding boxes in [x, y, w, h] format

    Returns:
        Bounding boxes in [x1, y1, x2, y2] format
    """
    if len(boxes) == 0:
        return boxes

    converted = boxes.copy()
    converted[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x + w
    converted[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y + h
    return converted


def clip_boxes(boxes: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Clip bounding boxes to image boundaries.

    Args:
        boxes: Bounding boxes in xyxy format
        image_shape: Image shape (height, width)

    Returns:
        Clipped bounding boxes
    """
    if len(boxes) == 0:
        return boxes

    h, w = image_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)  # y1, y2

    return boxes


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]

    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def non_max_suppression(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45
) -> List[int]:
    """
    Perform Non-Maximum Suppression on bounding boxes.
    Note: YOLO26 uses NMS-free inference, but this is kept for compatibility.

    Args:
        boxes: Bounding boxes in xyxy format
        scores: Confidence scores for each box
        iou_threshold: IoU threshold for suppression

    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []

    # Sort by scores (descending)
    indices = np.argsort(scores)[::-1]
    keep = []

    while len(indices) > 0:
        # Keep the box with highest score
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        # Calculate IoU with remaining boxes
        remaining = indices[1:]
        ious = np.array([calculate_iou(boxes[current], boxes[i]) for i in remaining])

        # Keep boxes with IoU below threshold
        indices = remaining[ious < iou_threshold]

    return keep


def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    class_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """
    Draw detections on image with YOLO26 optimization indicators.

    Args:
        image: Input image (BGR format)
        detections: List of detection dictionaries
        class_colors: Optional color mapping for classes

    Returns:
        Image with drawn detections
    """
    if class_colors is None:
        # Default colors with YOLO26 theme
        class_colors = {
            "person": (0, 255, 0),  # Green
            "sports ball": (255, 100, 0),  # Orange (YOLO26 accent)
        }

    annotated = image.copy()

    for detection in detections:
        bbox = detection["bbox"]
        conf = detection["confidence"]
        class_name = detection["class_name"]

        x1, y1, x2, y2 = map(int, bbox)
        color = class_colors.get(class_name, (255, 100, 0))  # Default orange

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{class_name}: {conf:.2f}"
        if detection.get("nms_free", False):
            label += " ✓"  # Indicate NMS-free processing

        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

        # Background for label
        cv2.rectangle(
            annotated,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            color,
            -1,
        )

        # Text
        cv2.putText(
            annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
        )

    return annotated


def validate_image_format(image: np.ndarray) -> bool:
    """
    Validate that image is in correct format for YOLO26.

    Args:
        image: Input image to validate

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(image, np.ndarray):
        logger.error("Image must be a numpy array")
        return False

    if len(image.shape) != 3:
        logger.error(f"Image must have 3 dimensions, got {len(image.shape)}")
        return False

    if image.shape[2] != 3:
        logger.error(f"Image must have 3 channels, got {image.shape[2]}")
        return False

    return True


def get_model_memory_usage(model_info: Dict[str, Any]) -> str:
    """
    Get human-readable model memory usage information with YOLO26 benefits.

    Args:
        model_info: Model information dictionary

    Returns:
        Formatted memory usage string
    """
    params = model_info.get("params", "Unknown")
    flops = model_info.get("flops", "Unknown")
    cpu_ms = model_info.get("cpu_ms", "Unknown")

    return f"Parameters: {params}, FLOPs: {flops}, CPU(ms): {cpu_ms}"


def log_detection_summary(
    detections: List[Dict[str, Any]], nms_free: bool = True
) -> None:
    """
    Log a summary of detections with YOLO26-specific information.

    Args:
        detections: List of detection dictionaries
        nms_free: Whether NMS-free inference was used
    """
    if not detections:
        logger.info("No detections found")
        return

    class_counts = {}
    total_confidence = 0.0

    for det in detections:
        class_name = det["class_name"]
        confidence = det["confidence"]

        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        total_confidence += confidence

    avg_confidence = total_confidence / len(detections)

    mode_str = "NMS-free" if nms_free else "Traditional NMS"
    logger.info(
        f"Found {len(detections)} objects (avg confidence: {avg_confidence:.3f}) [{mode_str}]"
    )

    for class_name, count in class_counts.items():
        logger.info(f"  {class_name}: {count}")


def optimize_inference_settings(hardware_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimize inference settings based on hardware capabilities with YOLO26 optimizations.

    Args:
        hardware_info: Hardware information dictionary

    Returns:
        Optimized settings dictionary
    """
    settings = {
        "half_precision": False,
        "batch_size": 1,
        "num_workers": 0,
        "nms_free": True,  # YOLO26 default
        "small_object_optimized": True,  # YOLO26 STAL
    }

    # GPU optimizations
    if hardware_info.get("cuda_available", False):
        settings["half_precision"] = True
        if hardware_info.get("gpu_memory_gb", 0) > 8:
            settings["batch_size"] = 2

    # CPU optimizations - YOLO26 excels here
    if hardware_info.get("cpu_cores", 1) >= 4:
        settings["num_workers"] = 2
        settings["nms_free"] = True  # Major CPU advantage

    return settings


def validate_nms_free_compatibility(model_config: Dict[str, Any]) -> bool:
    """
    Validate that configuration supports NMS-free inference.

    Args:
        model_config: Model configuration dictionary

    Returns:
        True if NMS-free compatible, False otherwise
    """
    # YOLO26 always supports NMS-free inference
    if model_config.get("model_variant", "").startswith("yolo26"):
        return True

    return False


def get_performance_summary(
    model_variant: str, nms_free: bool = True
) -> Dict[str, Any]:
    """
    Get performance summary for YOLO26 model variant.

    Args:
        model_variant: YOLO26 model variant
        nms_free: Whether NMS-free inference is enabled

    Returns:
        Performance summary dictionary
    """
    base_specs = {
        "yolo26n": {"cpu_speedup": "43%", "accuracy": "40.9 mAP"},
        "yolo26s": {"cpu_speedup": "41%", "accuracy": "48.6 mAP"},
        "yolo26m": {"cpu_speedup": "38%", "accuracy": "52.3 mAP"},
        "yolo26l": {"cpu_speedup": "35%", "accuracy": "54.7 mAP"},
        "yolo26x": {"cpu_speedup": "32%", "accuracy": "56.1 mAP"},
    }

    specs = base_specs.get(model_variant, {})

    if nms_free:
        specs["latency_reduction"] = "15-20%"
        specs["processing_mode"] = "End-to-end NMS-free"
    else:
        specs["processing_mode"] = "Traditional NMS"

    return specs
