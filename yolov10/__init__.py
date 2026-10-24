"""
YOLOv10 integration module for UltimatePlayerTracker.

This module provides YOLOv10 model management, inference, and utilities
for seamless integration with the existing DeepSORT tracking pipeline.
"""

from .config import YOLOv10Config
from .detection_adapter import DetectionAdapter
from .inference import YOLOv10Inference
from .model_loader import ModelLoader
from .utils import *

__version__ = "1.0.0"
__all__ = [
    "ModelLoader",
    "DetectionAdapter",
    "YOLOv10Config",
    "YOLOv10Inference",
]
