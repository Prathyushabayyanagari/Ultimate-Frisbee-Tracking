# Development Environment Setup Guide

This guide covers setting up the development environment for UltimatePlayerTracker's YOLO migration using the pixi package manager and modern tooling.

## Current Environment Analysis

The existing project uses pixi with outdated dependencies:
- **TensorFlow 2.3.0** (July 2020)
- **OpenCV 4.1.0** (outdated)
- **Python 3.8** (as specified in pyproject.toml)
- **Ruff** for linting and formatting
- **MyPy** for type checking

## Prerequisites

### System Requirements
- **Python**: 3.8 (as specified in pyproject.toml)
- **OS**: Linux-64 (current target), with plans for cross-platform support
- **Hardware**: NVIDIA GPU (recommended) or CPU for development
- **Memory**: 8GB+ RAM for model training and inference

### Required Tools
- **Pixi Package Manager**: Modern Python package management
- **Git**: Version control
- **CUDA**: 11.0+ (for GPU acceleration, optional)

## Environment Setup

### 1. Install Pixi Package Manager

```bash
# Install Pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Restart shell or source profile
source ~/.bashrc
```

### 2. Clone Repository and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd UltimatePlayerTracker

# Create pixi environment and install dependencies
pixi install

# Verify installation
pixi run python --version
pixi run ruff --version
pixi run mypy --version
```

### 3. Current Environment Configuration

The project uses pixi with the following configuration:

```toml
# pyproject.toml (current)
[tool.pixi.dependencies]
python = ">=3.8,<3.9"
numpy = ">=1.16.0,<1.19.0"
matplotlib = ">=3.4.0"
pandas = ">=1.2.0"
pillow = ">=8.4.0"
mypy = ">=0.910"
protobuf = "<=3.20.3"
scipy = ">=1.5.0"

[tool.pixi.pypi-dependencies]
tensorflow-cpu = "~=2.3.0"
opencv-python = "~=4.1.0"
tqdm = ">=4.62.0"
lxml = ">=4.6.0"
absl-py = ">=0.15.0"
easydict = ">=1.9"
ruff = "*"
```

### 4. Development Tools Configuration

#### Ruff (Linting & Formatting)
```bash
# Format code
pixi run ruff format .

# Run linting
pixi run ruff check .

# Fix auto-fixable issues
pixi run ruff check --fix .

# Run all quality checks
pixi run check-all
```

#### MyPy (Type Checking)
```bash
# Run type checking
pixi run mypy .
```

## YOLO Migration Environment Setup

### 1. Updated pyproject.toml for Migration

```toml
[tool.pixi.pypi-dependencies]
# Replace TensorFlow with PyTorch
torch = ">=2.0.0"
torchvision = ">=0.15.0"
ultralytics = ">=8.0.0"

# Updated OpenCV
opencv-python = ">=4.8.0"

# Keep existing dependencies
numpy = ">=1.16.0,<1.19.0"
matplotlib = ">=3.4.0"
pandas = ">=1.2.0"
pillow = ">=8.4.0"
tqdm = ">=4.62.0"
lxml = ">=4.6.0"
absl-py = ">=0.15.0"
easydict = ">=1.9"

# Development tools
ruff = "*"
mypy = ">=0.910"
pytest = ">=7.0.0"
```

### 2. Install Updated Dependencies

```bash
# Update dependencies in pixi environment
pixi install

# Verify new installations
pixi run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
pixi run python -c "from ultralytics import YOLO; print('Ultralytics installed successfully')"
```

### 3. Verify Installation

#### Test PyTorch Installation
```python
# test_pytorch.py
import torch
import torchvision

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

#### Test Ultralytics Installation
```python
# test_ultralytics.py
from ultralytics import YOLO

# Test model loading (will download default model)
model = YOLO('yolov10n.pt')
print(f"YOLOv10 model loaded successfully")
print(f"Model info: {model.info()}")

# Test inference on dummy image
import numpy as np
dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
results = model(dummy_image)
print(f"Inference test completed")
print(f"Detected {len(results[0].boxes)} objects")
```

#### Test OpenCV Installation
```python
# test_opencv.py
import cv2
import numpy as np

print(f"OpenCV version: {cv2.__version__}")

# Test basic operations
dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
resized = cv2.resize(dummy_image, (640, 640))
print(f"OpenCV resize test passed: {resized.shape}")
```

### 4. GPU Setup (if using NVIDIA GPU)

#### CUDA Toolkit Verification
```bash
# Check CUDA version
nvidia-smi

# Check if CUDA version matches PyTorch
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"
```

#### CuDNN Verification
```python
# test_cudnn.py
import torch

if torch.backends.cudnn.is_available():
    print(f"CuDNN version: {torch.backends.cudnn.version()}")
    print(f"CuDNN enabled: {torch.backends.cudnn.enabled}")
else:
    print("CuDNN not available")
```

### 5. Hardware-Specific Optimizations

#### For NVIDIA GPU Users
```bash
# Install TensorRT (optional, for deployment optimization)
# Download from NVIDIA developer portal
pip install tensorrt

# Test TensorRT availability
python -c "import tensorrt; print('TensorRT available')"
```

#### For Apple Silicon Users
```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Test MPS availability
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

#### For CPU-Only Users
```bash
# Install ONNX Runtime for CPU optimization
pip install onnxruntime

# Test ONNX Runtime
python -c "import onnxruntime; print('ONNX Runtime available')"
```

## Environment Variables

### Set for GPU Usage
```bash
# Linux/Mac
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Windows
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### Set for Memory Optimization
```bash
# Limit PyTorch memory usage (adjust based on your GPU memory)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Enable memory pooling for better performance
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8
```

## Development IDE Setup

### VS Code Extensions
- **Python** - Microsoft
- **PyTorch** - PyTorch team
- **Jupyter** - Microsoft
- **Remote Development** - Microsoft (if using remote GPU)

### PyCharm Setup
1. File → Settings → Project → Python Interpreter
2. Add new interpreter (point to virtual environment)
3. Install PyTorch plugin if desired

## Validation Checklist

### Basic Functionality Tests
- [ ] Python 3.8+ installed and working
- [ ] PyTorch 2.0+ loads without errors
- [ ] CUDA (if GPU) available and functional
- [ ] Ultralytics YOLO loads models successfully
- [ ] OpenCV operations work correctly
- [ ] Basic inference pipeline runs without errors

### Performance Validation
- [ ] GPU memory allocation works correctly
- [ ] CPU inference performance is acceptable
- [ ] Multi-threading (if applicable) works

### Integration Tests
- [ ] Can load sample video frames
- [ ] Can run basic object detection
- [ ] Can process detection results
- [ ] No import errors or missing dependencies

## Troubleshooting

### Common Issues

#### 1. CUDA Version Mismatch
```bash
# Symptoms: CUDA out of memory, import errors
# Solution: Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. OpenCV Import Errors
```bash
# Symptoms: cv2 import fails
# Solution: Install opencv-python instead of opencv-contrib-python
pip uninstall opencv-contrib-python
pip install opencv-python>=4.8.0
```

#### 3. Ultralytics Model Download Issues
```python
# Symptoms: Model download fails or times out
# Solution: Download models manually or set cache directory
import os
os.environ['ULTRALYTICS_CACHE'] = '/path/to/cache/directory'
```

#### 4. Memory Issues on Limited Hardware
```python
# Solutions for low-memory systems:
# 1. Use smaller model variants
model = YOLO('yolov10n.pt')  # Use 'n' variant

# 2. Reduce batch size
results = model(image, batch=1)

# 3. Use CPU fallback if needed
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 4000000000:
    device = 'cpu'
else:
    device = 'cuda'
```

## Development Workflow

### 1. Code Quality Pipeline

```bash
# Format code before commits
pixi run ruff format .

# Run linting
pixi run ruff check .

# Fix issues automatically
pixi run ruff check --fix .

# Run type checking
pixi run mypy .

# Run all checks together
pixi run check-all
```

### 2. Testing YOLO Models

```bash
# Test YOLOv10 model loading
pixi run python -c "
from ultralytics import YOLO
model = YOLO('yolov10s.pt')
print(f'Model loaded: {model.names}')
"

# Test inference on sample image
pixi run python -c "
import cv2
from ultralytics import YOLO

model = YOLO('yolov10s.pt')
image = cv2.imread('data/video/demo_frame.jpg')
results = model(image)
print(f'Inference successful: {len(results[0].boxes)} detections')
"
```

### 3. Performance Benchmarking

```bash
# Benchmark YOLOv10 performance
pixi run python -c "
import time
from ultralytics import YOLO

model = YOLO('yolov10s.pt')
image = 'data/video/demo_frame.jpg'

# Warm up
model(image)

# Benchmark
start_time = time.time()
for _ in range(100):
    model(image)
end_time = time.time()

avg_time = (end_time - start_time) / 100
fps = 1.0 / avg_time
print(f'Average inference time: {avg_time:.3f}s')
print(f'Estimated FPS: {fps:.1f}')
"
```

## Environment Management

### 1. Environment Commands

```bash
# List all environments
pixi list

# Switch to migration environment
pixi shell yolov10-migration

# Remove environment (if needed)
pixi remove yolov10-migration
```

### 2. Dependency Management

```bash
# Add new dependency
pixi add <package-name>

# Add development dependency
pixi add --dev <package-name>

# Remove dependency
pixi remove <package-name>

# Update dependencies
pixi update
```

## Validation Checklist

### Basic Functionality Tests
- [ ] Python 3.8+ installed and working
- [ ] PyTorch 2.0+ loads without errors
- [ ] CUDA (if GPU) available and functional
- [ ] Ultralytics YOLO loads models successfully
- [ ] OpenCV operations work correctly
- [ ] Basic inference pipeline runs without errors
- [ ] Ruff formatting and linting work
- [ ] MyPy type checking works

### Performance Validation
- [ ] GPU memory allocation works correctly
- [ ] CPU inference performance is acceptable
- [ ] YOLOv10 model loads and runs inference
- [ ] Code quality tools function properly

### Integration Tests
- [ ] Can load sample video frames
- [ ] Can run basic object detection
- [ ] Can process detection results
- [ ] No import errors or missing dependencies

## Next Steps

After completing environment setup:

1. **Run all validation tests** above
2. **Document any issues** in Project/environment-issues.md
3. **Begin Phase 2** of the migration plan
4. **Update progress tracker** with environment setup completion

## Support Resources

- **Ultralytics Documentation**: https://docs.ultralytics.com/
- **PyTorch Installation**: https://pytorch.org/get-started/locally/
- **Pixi Documentation**: https://pixi.sh/
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-toolkit
- **Project Issues**: Create GitHub issues for blocking problems

---

*Last Updated: January 15, 2026*
*Updated for pixi-based workflow*