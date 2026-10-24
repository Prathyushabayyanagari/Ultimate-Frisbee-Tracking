"""
YOLO26 Configuration

Configuration management for YOLO26 models and detection parameters.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class YOLO26Config:
    """
    Configuration class for YOLO26 detection parameters.

    This class provides a centralized configuration system for YOLO26 models,
    supporting different variants, hardware targets, and optimization settings.
    """

    # Model Configuration
    model_variant: str = "yolo26s"  # n, s, m, l, x
    model_path: Optional[str] = None  # Custom model path
    confidence_threshold: float = 0.30  # Optimized for YOLO26
    iou_threshold: float = 0.45

    # Hardware Configuration
    device: str = "auto"  # auto, cpu, cuda, mps
    batch_size: int = 1
    num_workers: int = 0

    # Image Configuration
    input_size: int = 640  # Square input size
    letterbox: bool = True  # Maintain aspect ratio

    # Performance Configuration
    half_precision: bool = True  # FP16 for GPU
    optimize_for_speed: bool = True  # Optimize for inference speed
    max_det: int = 300  # Maximum detections per image

    # YOLO26-specific Configuration
    nms_free: bool = True  # Native NMS-free inference
    small_object_optimized: bool = True  # STAL small-target-aware learning
    progressive_loss: bool = True  # ProgLoss for better accuracy

    # Class Filtering (for Ultimate frisbee)
    target_classes: List[int] = None  # None = all classes
    class_names: Dict[int, str] = None

    # Export Configuration
    export_format: str = "onnx"  # onnx, torchscript, coreml, tflite
    export_optimize: bool = True

    # Tracking Integration
    deepsort_compatible: bool = True  # Ensure DeepSORT compatibility
    coordinate_format: str = "xyxy"  # xyxy, xywh

    # Logging and Debugging
    verbose: bool = False
    save_results: bool = False
    output_dir: str = "./outputs"

    def __post_init__(self):
        """Initialize default values after creation."""
        if self.target_classes is None:
            # Default to person (0) and sports ball (37) for Ultimate frisbee
            self.target_classes = [0, 37]

        if self.class_names is None:
            # COCO class names (subset relevant for Ultimate frisbee)
            self.class_names = {
                0: "person",
                32: "sports ball",  # Updated for COCO
                37: "sports ball",  # Alternative index
            }

        # Auto-detect device if needed
        if self.device == "auto":
            self.device = self._detect_device()

    def _detect_device(self) -> str:
        """Automatically detect the best available device."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except ImportError:
            return "cpu"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "YOLO26Config":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_args(cls, args) -> "YOLO26Config":
        """Create configuration from command line arguments."""
        config_dict = {}

        # Map common argument names
        arg_mapping = {
            "model": "model_variant",
            "weights": "model_path",
            "score": "confidence_threshold",
            "iou": "iou_threshold",
            "device": "device",
            "img_size": "input_size",
            "half": "half_precision",
            "batch_size": "batch_size",
        }

        for arg_name, config_name in arg_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                config_dict[config_name] = getattr(args, arg_name)

        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model_variant": self.model_variant,
            "model_path": self.model_path,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
            "batch_size": self.batch_size,
            "input_size": self.input_size,
            "half_precision": self.half_precision,
            "nms_free": self.nms_free,
            "small_object_optimized": self.small_object_optimized,
            "progressive_loss": self.progressive_loss,
            "target_classes": self.target_classes,
            "export_format": self.export_format,
            "verbose": self.verbose,
        }

    def validate(self) -> bool:
        """Validate configuration parameters."""
        errors = []

        # Validate model variant
        valid_variants = [
            "yolo26n",
            "yolo26s",
            "yolo26m",
            "yolo26l",
            "yolo26x",
        ]
        if self.model_variant not in valid_variants:
            errors.append(f"Invalid model variant: {self.model_variant}")

        # Validate thresholds
        if not 0.0 <= self.confidence_threshold <= 1.0:
            errors.append(f"Invalid confidence threshold: {self.confidence_threshold}")

        if not 0.0 <= self.iou_threshold <= 1.0:
            errors.append(f"Invalid IOU threshold: {self.iou_threshold}")

        # Validate input size
        if self.input_size <= 0 or self.input_size > 1280:
            errors.append(f"Invalid input size: {self.input_size}")

        # Validate batch size
        if self.batch_size <= 0:
            errors.append(f"Invalid batch size: {self.batch_size}")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

        return True

    def get_model_info(self) -> Dict[str, Any]:
        """Get model variant information."""
        model_specs = {
            "yolo26n": {
                "params": "2.4M",
                "flops": "5.8B",
                "map": "40.9",
                "cpu_ms": "38.9",
            },
            "yolo26s": {
                "params": "9.5M",
                "flops": "18.2B",
                "map": "48.6",
                "cpu_ms": "87.2",
            },
            "yolo26m": {
                "params": "15.4M",
                "flops": "28.7B",
                "map": "52.3",
                "cpu_ms": "142.1",
            },
            "yolo26l": {
                "params": "25.8M",
                "flops": "46.3B",
                "map": "54.7",
                "cpu_ms": "238.4",
            },
            "yolo26x": {
                "params": "54.2M",
                "flops": "93.8B",
                "map": "56.1",
                "cpu_ms": "489.7",
            },
        }

        return model_specs.get(self.model_variant, {})

    def optimize_for_hardware(self) -> "YOLO26Config":
        """Optimize configuration based on detected hardware."""
        if self.device == "cpu":
            # CPU optimizations - YOLO26 excels here
            self.half_precision = False
            self.optimize_for_speed = True
            self.batch_size = 1
            self.nms_free = True  # NMS-free is major CPU advantage
            # YOLO26 is so efficient we can use larger models on CPU
            if self.model_variant in ["yolo26x"]:
                self.model_variant = "yolo26l"

        elif self.device == "cuda":
            # GPU optimizations
            self.half_precision = True
            self.optimize_for_speed = True
            self.nms_free = True
            # Can use larger models on GPU
            if self.model_variant == "yolo26n":
                self.model_variant = "yolo26s"

        return self

    def copy(self, **kwargs) -> "YOLO26Config":
        """Create a copy of the configuration with optional updates."""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return YOLO26Config.from_dict(config_dict)


# Default configuration instance
DEFAULT_CONFIG = YOLO26Config()
