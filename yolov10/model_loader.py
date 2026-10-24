"""
YOLOv10 Model Loader

Handles loading and management of YOLOv10 models with support for
different variants and optimization targets.
"""

import logging
import warnings
from typing import Any, Dict, Optional

import torch
from ultralytics import YOLO

# Suppress ultralytics torch.load FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="ultralytics.*")

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    YOLOv10 model loader with support for different variants and optimizations.

    This class handles loading YOLOv10 models in various formats and configurations,
    providing a unified interface for model management.
    """

    # Supported model variants
    MODEL_VARIANTS = {
        "yolov10n": "yolov10n.pt",  # Nano - smallest, fastest
        "yolov10s": "yolov10s.pt",  # Small - balanced
        "yolov10m": "yolov10m.pt",  # Medium - higher accuracy
        "yolov10b": "yolov10b.pt",  # Large
        "yolov10l": "yolov10l.pt",  # Large
        "yolov10x": "yolov10x.pt",  # Extra Large - most accurate
    }

    def __init__(
        self,
        model_variant: str = "yolov10s",
        device: Optional[str] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        """
        Initialize YOLOv10 model loader.

        Args:
            model_variant: Model variant name (n, s, m, b, l, x)
            device: Target device ('cuda', 'cpu', 'mps', or None for auto)
            confidence_threshold: Detection confidence threshold
            iou_threshold: NMS IOU threshold
        """
        self.model_variant = model_variant
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device or self._get_default_device()
        self.model: Optional[YOLO] = None

        # Validate model variant
        if model_variant not in self.MODEL_VARIANTS:
            raise ValueError(
                f"Unsupported model variant: {model_variant}. "
                f"Supported: {list(self.MODEL_VARIANTS.keys())}"
            )

    def _get_default_device(self) -> str:
        """Automatically determine the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def load_model(self, model_path: Optional[str] = None) -> YOLO:
        """
        Load YOLOv10 model.

        Args:
            model_path: Custom model path, if not using pretrained variant

        Returns:
            Loaded YOLO model
        """
        if model_path:
            model_file = model_path
        else:
            model_file = self.MODEL_VARIANTS[self.model_variant]

        try:
            logger.info(f"Loading YOLOv10 model: {model_file}")
            self.model = YOLO(model_file)

            # Move model to specified device
            self.model.to(self.device)

            logger.info(f"Model loaded successfully on {self.device}")
            logger.info(f"Model info: {self.model.info()}")

            return self.model

        except Exception as e:
            logger.error(f"Failed to load model {model_file}: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary containing model information
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        return {
            "variant": self.model_variant,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "model_names": self.model.names,
            "model_info": self.model.info(),
        }

    def optimize_for_inference(self) -> None:
        """
        Optimize the loaded model for inference.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Set model to evaluation mode
        self.model.model.eval()

        # Enable optimizations
        if self.device == "cuda":
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        logger.info("Model optimized for inference")

    def export_model(
        self, export_format: str = "onnx", export_path: Optional[str] = None
    ) -> str:
        """
        Export model to specified format.

        Args:
            export_format: Export format ('onnx', 'torchscript', 'coreml', 'tflite')
            export_path: Custom export path

        Returns:
            Path to exported model
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if export_path is None:
            export_path = f"{self.model_variant}_{export_format}"

        try:
            logger.info(f"Exporting model to {export_format}: {export_path}")

            # Export using Ultralytics export functionality
            exported_path = self.model.export(
                format=export_format,
                imgsz=640,
                optimize=True,
                half=True if self.device == "cuda" else False,
            )

            logger.info(f"Model exported successfully: {exported_path}")
            return str(exported_path)

        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            raise

    def update_thresholds(
        self, confidence: Optional[float] = None, iou: Optional[float] = None
    ) -> None:
        """
        Update detection thresholds.

        Args:
            confidence: New confidence threshold
            iou: New IOU threshold
        """
        if confidence is not None:
            self.confidence_threshold = max(0.0, min(1.0, confidence))

        if iou is not None:
            self.iou_threshold = max(0.0, min(1.0, iou))

        logger.info(
            f"Updated thresholds - confidence: {self.confidence_threshold}, "
            f"iou: {self.iou_threshold}"
        )

    def get_supported_variants(self) -> Dict[str, str]:
        """
        Get all supported model variants.

        Returns:
            Dictionary mapping variant names to descriptions
        """
        return {
            "yolov10n": "Nano - Smallest and fastest (2.3M params)",
            "yolov10s": "Small - Balanced performance (7.2M params)",
            "yolov10m": "Medium - Higher accuracy (15.4M params)",
            "yolov10b": "Base - Good accuracy (24.4M params)",
            "yolov10l": "Large - High accuracy (29.5M params)",
            "yolov10x": "Extra Large - Highest accuracy (56.9M params)",
        }
