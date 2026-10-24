"""
YOLOv10 Configuration
Configuration management for detection and tracking.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class YOLOv10Config:
    """Configuration for YOLOv10 detection."""

    model_variant: str = "yolov10s"
    model_path: Optional[str] = None
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45

    device: str = "auto"
    batch_size: int = 1
    num_workers: int = 0

    input_size: int = 1280
    letterbox: bool = True

    half_precision: bool = False
    optimize_for_speed: bool = True
    max_det: int = 300

    target_classes: Optional[List[int]] = None
    class_names: Optional[Dict[int, str]] = None

    export_format: str = "onnx"
    export_optimize: bool = True

    deepsort_compatible: bool = True
    coordinate_format: str = "xyxy"

    verbose: bool = False
    save_results: bool = False
    output_dir: str = "./outputs"

    def __post_init__(self):
        if self.target_classes is None:
            self.target_classes = [0, 29, 32]

        if self.class_names is None:
            self.class_names = {
                0: "person",
                29: "frisbee",
                32: "sports ball",
            }

        if self.device == "auto":
            self.device = self._detect_device()

    def _detect_device(self) -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"

            if (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                return "mps"

        except ImportError:
            pass

        return "cpu"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "YOLOv10Config":
        return cls(**config_dict)

    @classmethod
    def from_args(cls, args) -> "YOLOv10Config":
        config_dict = {}

        arg_mapping = {
            "model": "model_variant",
            "weights": "model_path",
            "score": "confidence_threshold",
            "iou": "iou_threshold",
            "device": "device",
            "img_size": "input_size",
            "size": "input_size",
            "half": "half_precision",
            "batch_size": "batch_size",
        }

        for arg_name, config_name in arg_mapping.items():
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)

                if value is not None:
                    config_dict[config_name] = value

        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_variant": self.model_variant,
            "model_path": self.model_path,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
            "batch_size": self.batch_size,
            "input_size": self.input_size,
            "half_precision": self.half_precision,
            "target_classes": self.target_classes,
            "class_names": self.class_names,
            "export_format": self.export_format,
            "verbose": self.verbose,
            "max_det": self.max_det,
        }

    def validate(self) -> bool:
        errors = []

        valid_variants = [
            "yolov10n",
            "yolov10s",
            "yolov10m",
            "yolov10b",
            "yolov10l",
            "yolov10x",
        ]

        if self.model_variant not in valid_variants:
            errors.append(
                f"Invalid model variant: {self.model_variant}"
            )

        if not 0.0 <= self.confidence_threshold <= 1.0:
            errors.append("Confidence threshold must be between 0 and 1.")

        if not 0.0 <= self.iou_threshold <= 1.0:
            errors.append("IoU threshold must be between 0 and 1.")

        if not 1 <= self.input_size <= 1280:
            errors.append("Input size must be between 1 and 1280.")

        if self.batch_size <= 0:
            errors.append("Batch size must be positive.")

        if errors:
            raise ValueError(
                "Configuration validation failed:\n"
                + "\n".join(errors)
            )

        return True

    def get_model_info(self) -> Dict[str, Any]:
        model_specs = {
            "yolov10n": {"params": "2.3M", "flops": "6.7B", "map": "39.5"},
            "yolov10s": {"params": "7.2M", "flops": "21.6B", "map": "46.7"},
            "yolov10m": {"params": "15.4M", "flops": "59.1B", "map": "51.3"},
            "yolov10b": {"params": "24.4M", "flops": "92.0B", "map": "52.7"},
            "yolov10l": {"params": "29.5M", "flops": "120.3B", "map": "53.3"},
            "yolov10x": {"params": "56.9M", "flops": "160.4B", "map": "54.4"},
        }

        return model_specs.get(self.model_variant, {})

    def optimize_for_hardware(self) -> "YOLOv10Config":
        if self.device == "cpu":
            self.half_precision = False
            self.optimize_for_speed = True
            self.batch_size = 1

        elif self.device == "cuda":
            self.half_precision = True
            self.optimize_for_speed = True

        return self

    def copy(self, **kwargs) -> "YOLOv10Config":
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return YOLOv10Config.from_dict(config_dict)


DEFAULT_CONFIG = YOLOv10Config()
