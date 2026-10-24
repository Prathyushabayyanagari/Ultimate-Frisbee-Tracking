"""
YOLO26 integration module for UltimatePlayerTracker.

This module provides YOLO26 model management, inference, and utilities
for seamless integration with the existing DeepSORT tracking pipeline.
"""

from .config import YOLO26Config
from .detection_adapter import DetectionAdapter
from .inference import YOLO26Inference
from .model_loader import ModelLoader
from .utils import *

__version__ = "1.0.0"
__all__ = [
    "ModelLoader",
    "DetectionAdapter",
    "YOLO26Config",
    "YOLO26Inference",
]
