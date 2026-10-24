# AGENTS.md

This file contains guidelines and commands for agentic coding agents working in the UltimatePlayerTracker repository.

## Project Overview

UltimatePlayerTracker is a Python-based object tracking system for Ultimate frisbee players using YOLOv10 for object detection and DeepSort for tracking. The project uses PyTorch/Ultralytics and follows a modern computer vision pipeline architecture. Legacy YOLOv4/TensorFlow support is maintained for comparison purposes.

## Build/Lint/Test Commands

### Environment Setup
```bash
# Install Pixi package manager (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Install dependencies with Pixi
pixi install

# Or install with pip directly (legacy)
pip install -r requirements.txt  # For CPU
pip install -r requirements-gpu.txt  # For GPU
```

### Code Quality Tools
The project uses the following development dependencies (defined in pyproject.toml):
- **ruff**: Combined linting and formatting (10-100x faster than black/flake8)
- **mypy**: Type checking (v0.910)

```bash
# Format code
pixi run ruff format .

# Run linting
pixi run ruff check .

# Fix auto-fixable issues
pixi run ruff check --fix .

# Run type checking
pixi run mypy .

# Run all quality checks together
pixi run check-all
```

### Running the Application
```bash
# Run object tracker with YOLOv10 (default)
pixi run python object_tracker.py --video ./data/video/demo.mp4 --output ./outputs/demo.avi --model yolov10

# Download and export YOLOv10 model
pixi run python save_model.py --model yolov10s --export_format onnx

# Run with webcam (set video flag to 0)
pixi run python object_tracker.py --video 0 --output ./outputs/webcam.avi --model yolov10

# Run with specific YOLOv10 variant
pixi run python object_tracker.py --video ./data/video/test.mp4 --model yolov10n --device cuda

# Run with custom parameters
pixi run python object_tracker.py --video ./data/video/demo.mp4 --output ./outputs/demo.avi --model yolov10s --confidence 0.3 --iou 0.5

# Legacy YOLOv4 support
pixi run python object_tracker.py --video ./data/video/demo.mp4 --output ./outputs/demo.avi --model yolov4
```

### Testing
This project currently does not have formal unit tests. Testing is done by running the main application with sample videos. When adding tests, place them in a `tests/` directory and use pytest.

## Code Style Guidelines

### Import Organization
- Standard library imports first (os, sys, time, etc.)
- Third-party imports next (numpy, torch, cv2, PIL, etc.)
- Local imports last (core.*, deep_sort.*, yolov10.*, tools.*)
- Use absolute imports for local modules
- Group related imports together

```python
# Standard library
import os
import time

# Third-party
import numpy as np
import torch
import cv2
from PIL import Image

# Local imports
import core.utils as utils
from yolov10.model_loader import ModelLoader
from yolov10.inference import YOLOv10Inference
from yolov10.detection_adapter import DetectionAdapter
from deep_sort.tracker import Tracker

# Legacy YOLOv4 imports (when needed)
# from core.yolov4 import filter_boxes
```

### Formatting Conventions
- Use Ruff for automatic formatting (line length 88)
- 4-space indentation (no tabs)
- Maximum line length: 88 characters
- Use double quotes for strings and docstrings
- Leave 2 blank lines before top-level functions/classes

### Type Hints
- Use type hints for function parameters and return values
- Import typing constructs as needed (Optional, List, Tuple, etc.)
- Use numpy array types where appropriate: `np.ndarray`

### Naming Conventions
- **Variables**: snake_case (e.g., `max_cosine_distance`, `frame_size`)
- **Functions**: snake_case (e.g., `filter_boxes`, `load_weights`)
- **Classes**: PascalCase (e.g., `Tracker`, `Detection`, `KalmanFilter`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `FLAGS`, `TRANSFORM_MAT`)
- **Module names**: snake_case (e.g., `object_tracker.py`, `kalman_filter.py`)

### Error Handling
- Use try-except blocks for file I/O and video operations
- Handle TensorFlow GPU configuration gracefully
- Provide meaningful error messages
- Use specific exception types when possible

```python
try:
    vid = cv2.VideoCapture(int(video_path))
except:
    vid = cv2.VideoCapture(video_path)
```

### Documentation
- Use docstrings for all classes and public functions
- Follow NumPy docstring format for parameters and returns
- Include parameter types and descriptions
- Add brief usage examples for complex functions

### PyTorch Specific Guidelines
- Use PyTorch 2.0+ patterns with automatic mixed precision
- Configure device (CPU/CUDA/MPS) based on hardware availability
- Use torch.nn for model components when building custom modules
- Prefer Ultralytics patterns for YOLO model handling

### TensorFlow Specific Guidelines (Legacy)
- Use TensorFlow 2.3.0 for YOLOv4 legacy support only
- Configure GPU memory growth to prevent allocation issues
- Use tf.constant for fixed values, tf.Variable for trainable parameters
- Prefer tf.keras layers when building models

### Computer Vision Pipeline Patterns
- Follow the detection → tracking → visualization pipeline
- Use consistent coordinate systems (normalize when needed)
- Handle frame-by-frame processing in main loops
- Store intermediate results for debugging

### Configuration Management
- Use the cfg object from core.config for all configuration
- Define constants at module level when appropriate
- Use command-line flags (absl.flags) for runtime parameters

### Configuration Management
- Use the cfg object from core.config for all configuration
- Define constants at module level when appropriate
- Use command-line flags (absl.flags) for runtime parameters

### File Organization
- `core/`: YOLOv4 legacy implementation and utilities
- `deep_sort/`: Tracking algorithm components
- `tools/`: Helper scripts and utilities
- `yolov10/`: YOLOv10 implementation modules
  - `model_loader.py`: Model loading and management
  - `inference.py`: YOLOv10 inference engine
  - `detection_adapter.py`: DeepSORT compatibility layer
  - `config.py`: YOLOv10 configuration
- `data/`: Input data, weights, and class files
- `checkpoints/`: Saved TensorFlow models (legacy)

### Performance Considerations
- Minimize memory allocations in video processing loops
- Use numpy operations instead of Python loops when possible
- Pre-allocate arrays for repeated operations
- Consider GPU memory usage for TensorFlow operations
- For YOLOv10, leverage hardware auto-detection and optimization

### Testing and Validation
- Test with sample videos before deploying
- Validate model paths and file existence
- Check video capture properties before processing
- Log frame processing metrics (FPS) for performance monitoring

## Framework-Specific Notes

### TensorFlow 2.3.0
- This project uses an older TensorFlow version for compatibility
- Avoid using newer TF features not available in 2.3.0
- Use tf.compat.v1 for TF 1.x compatibility when needed

### OpenCV
- Use BGR format for OpenCV operations, convert to RGB when needed
- Handle video codec compatibility for output formats
- Use cv2.VideoWriter for output video generation

### DeepSort Integration
- Follow the existing DeepSort class interfaces
- Maintain compatibility with the original DeepSort paper implementation
- Use cosine distance for appearance-based matching