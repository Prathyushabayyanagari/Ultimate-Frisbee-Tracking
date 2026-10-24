
"""
YOLOv10 Inference Wrapper
High-resolution object detection with DeepSORT-compatible results.
"""

import logging
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class YOLOv10Inference:
    """Run YOLO inference and return detections in the project's format."""

    def __init__(self, model, config, person_confidence: float = 0.25):
        self.model = model
        self.config = config
        self.inference_count = 0
        self.total_inference_time = 0.0

        if not 0.0 <= person_confidence <= 1.0:
            raise ValueError("Person confidence must be between 0 and 1.")
        self.person_confidence = person_confidence
        self.disc_confidence = 0.01

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Return the original OpenCV BGR image."""
        if image is None or image.size == 0:
            raise ValueError("Input image is empty.")
        return image

    def _run_model(self, images):
        """Run inference with configured image size and confidence threshold."""
        return self.model.predict(
            source=images,
            imgsz=getattr(self.config, "input_size", 1280),
            conf=self.disc_confidence,
            iou=self.config.iou_threshold,
            max_det=self.config.max_det,
            device=getattr(self.config, "device", "cpu"),
            verbose=self.config.verbose,
        )

    def infer_single(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run inference on one BGR image."""
        processed_image = self.preprocess_image(image)
        start_time = time.time()

        try:
            results = self._run_model(processed_image)

            elapsed = time.time() - start_time
            self.inference_count += 1
            self.total_inference_time += elapsed

            if self.config.verbose:
                logger.debug("Inference time: %.3f seconds", elapsed)

            if not results:
                return []

            return self._process_results(results[0])

        except Exception:
            logger.exception("Single-image inference failed.")
            raise

    def infer_batch(
        self, images: List[np.ndarray]
    ) -> List[List[Dict[str, Any]]]:
        """Run inference on a batch of BGR images."""
        if not images:
            return []

        processed_images = [
            self.preprocess_image(image) for image in images
        ]
        start_time = time.time()

        try:
            results = self._run_model(processed_images)

            elapsed = time.time() - start_time
            self.inference_count += len(images)
            self.total_inference_time += elapsed

            if self.config.verbose:
                logger.debug(
                    "Batch inference: %.3f seconds for %d images",
                    elapsed,
                    len(images),
                )

            return [self._process_results(result) for result in results]

        except Exception:
            logger.exception("Batch inference failed.")
            raise

    def _process_results(self, result) -> List[Dict[str, Any]]:
        """
        Convert YOLO boxes to the existing detection dictionary format.

        COCO class IDs:
            0  = person
            29 = frisbee
            32 = sports ball
        """
        detections = []
        boxes = getattr(result, "boxes", None)

        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()

        configured_classes = getattr(
            self.config, "target_classes", None
        )
        class_names = getattr(self.model, "names", {})
        configured_names = set()
        configured_name_map = getattr(self.config, "class_names", None) or {}
        if configured_classes is not None:
            configured_names = {
                str(configured_name_map[class_id]).strip().lower()
                for class_id in configured_classes
                if class_id in configured_name_map
            }

        for box, confidence, class_value in zip(
            xyxy, confidences, class_ids
        ):
            class_id = int(class_value)
            confidence = float(confidence)

            if isinstance(class_names, dict):
                class_name = class_names.get(
                    class_id, f"class_{class_id}"
                )
            else:
                class_name = (
                    class_names[class_id]
                    if class_id < len(class_names)
                    else f"class_{class_id}"
                )

            normalized_name = str(class_name).strip().lower()
            if (
                configured_classes is not None
                and class_id not in configured_classes
                and normalized_name not in configured_names
            ):
                continue

            if normalized_name == "person" or (
                normalized_name.startswith("class_") and class_id == 0
            ):
                if confidence < self.person_confidence:
                    continue
            elif normalized_name in ("frisbee", "sports ball") or (
                normalized_name.startswith("class_")
                and class_id in (29, 32)
            ):
                if confidence < self.disc_confidence:
                    continue
            else:
                continue

            detections.append(
                {
                    "bbox": box.astype(float).tolist(),
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name,
                }
            )

        return detections

    def get_performance_stats(self) -> Dict[str, float]:
        """Return basic inference performance statistics."""
        if self.inference_count == 0:
            return {
                "avg_inference_time": 0.0,
                "fps": 0.0,
                "total_inferences": 0,
                "total_time": 0.0,
            }

        average = self.total_inference_time / self.inference_count

        return {
            "avg_inference_time": average,
            "fps": 1.0 / average if average > 0 else 0.0,
            "total_inferences": self.inference_count,
            "total_time": self.total_inference_time,
        }

    def reset_stats(self):
        """Reset inference performance counters."""
        self.inference_count = 0
        self.total_inference_time = 0.0

    def detect_video_frame(
        self, frame: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Detect objects and optionally annotate the input frame."""
        detections = self.infer_single(frame)
        annotated_frame = frame.copy()

        if self.config.verbose or self.config.save_results:
            annotated_frame = self._annotate_frame(
                annotated_frame, detections
            )

        return detections, annotated_frame

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Draw people in green and frisbee-class predictions in orange."""
        annotated = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = map(
                int, detection["bbox"]
            )
            confidence = detection["confidence"]
            class_id = detection["class_id"]
            class_name = detection["class_name"]

            if class_id != 0 and str(class_name).strip().lower() != "frisbee":
                continue

            # Green for people; orange only for frisbee-class detections.
            color = (0, 255, 0) if class_id == 0 else (0, 165, 255)

            cv2.rectangle(
                annotated, (x1, y1), (x2, y2), color, 2
            )

            label = f"{class_name}: {confidence:.2f}"
            text_y = max(20, y1 - 8)

            cv2.putText(
                annotated,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        return annotated
