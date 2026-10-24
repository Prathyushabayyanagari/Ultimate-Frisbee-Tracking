
"""
YOLOv10 Detection Adapter
Converts person and frisbee detections to DeepSORT format.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DetectionAdapter:
    """Convert YOLO detections into DeepSORT-compatible arrays."""

    def __init__(
        self,
        target_classes: Optional[List[str]] = None,
        confidence_threshold: float = 0.25,
    ):
        self.target_classes = [
            name.strip().lower()
            for name in (
                target_classes
                or ["person", "frisbee", "sports ball"]
            )
        ]

        self.confidence_threshold = confidence_threshold
        self.person_threshold = 0.25
        self.disc_threshold = 0.01

        self.class_mapping = {
            "person": 0,
            "frisbee": 29,
            "sports ball": 32,
        }

        self.reverse_mapping = {
            class_id: class_name
            for class_name, class_id in self.class_mapping.items()
        }

    def validate_detections(
        self, detections: List[Dict[str, Any]]
    ) -> bool:
        if not isinstance(detections, list):
            return False

        required = {"bbox", "confidence", "class_id", "class_name"}

        for detection in detections:
            if not isinstance(detection, dict):
                return False

            if not required.issubset(detection.keys()):
                return False

            bbox = detection["bbox"]

            if not isinstance(bbox, (list, tuple, np.ndarray)):
                return False

            if len(bbox) != 4:
                return False

        return True

    def filter_by_class(
        self,
        detections: List[Dict[str, Any]],
        allowed_classes: List[str],
    ) -> List[Dict[str, Any]]:
        allowed = {name.strip().lower() for name in allowed_classes}

        return [
            d for d in detections
            if str(d["class_name"]).strip().lower() in allowed
        ]

    def filter_ultimate_frisbee_objects(
        self, detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return self.filter_by_class(
            detections, ["person", "frisbee", "sports ball"]
        )

    def convert_to_deepsort_format(
        self,
        detections: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        boxes, scores, classes = [], [], []

        for detection in detections:
            name = str(detection["class_name"]).strip().lower()

            if name not in self.target_classes:
                continue

            try:
                confidence = float(detection["confidence"])
                x1, y1, x2, y2 = map(float, detection["bbox"])
                class_id = int(detection["class_id"])
            except (TypeError, ValueError, KeyError):
                logger.warning("Skipping malformed detection: %s", detection)
                continue

            if name == "person":
                threshold = self.person_threshold
            elif name in ("frisbee", "sports ball"):
                threshold = self.disc_threshold
            else:
                continue

            if confidence < threshold:
                continue

            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                continue

            boxes.append([x1, y1, width, height])
            scores.append(confidence)
            classes.append(class_id)

        if not boxes:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                0,
            )

        return (
            np.asarray(boxes, dtype=np.float32),
            np.asarray(scores, dtype=np.float32),
            np.asarray(classes, dtype=np.int32),
            len(boxes),
        )

    def format_for_deepsort(
        self,
        detections: List[Dict[str, Any]],
        image_shape: Tuple[int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        if not self.validate_detections(detections):
            logger.error("Invalid detections provided.")
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                0,
            )

        filtered = self.filter_ultimate_frisbee_objects(detections)
        return self.convert_to_deepsort_format(filtered)

    def convert_coordinates(
        self,
        bboxes_xyxy: np.ndarray,
        input_size: Tuple[int, int],
        original_size: Tuple[int, int],
    ) -> np.ndarray:
        if len(bboxes_xyxy) == 0:
            return bboxes_xyxy

        input_w, input_h = input_size
        original_w, original_h = original_size

        if min(input_w, input_h) <= 0:
            raise ValueError("Input dimensions must be positive.")

        scaled = bboxes_xyxy.copy().astype(np.float32)
        scaled[:, 0] *= original_w / input_w
        scaled[:, 1] *= original_h / input_h
        scaled[:, 2] *= original_w / input_w
        scaled[:, 3] *= original_h / input_h

        return scaled

    def get_class_statistics(
        self, detections: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        counts: Dict[str, int] = {}

        for detection in detections:
            name = str(detection.get("class_name", "unknown"))
            counts[name] = counts.get(name, 0) + 1

        return counts

    def get_info(self) -> Dict[str, Any]:
        return {
            "target_classes": self.target_classes,
            "confidence_threshold": self.confidence_threshold,
            "person_threshold": self.person_threshold,
            "disc_threshold": self.disc_threshold,
            "class_mapping": self.class_mapping,
            "reverse_mapping": self.reverse_mapping,
        }
