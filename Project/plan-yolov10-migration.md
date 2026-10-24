# UltimatePlayerTracker YOLO Migration Project Plan

## Project Overview

This project aims to migrate UltimatePlayerTracker from outdated YOLOv4 (TensorFlow 2.3.0, 2020) to a modern YOLO version (YOLOv10/YOLOv8) via Ultralytics framework. The migration will provide:

- **3-4x performance improvement** in FPS
- **5-10% accuracy improvement** for player/frisbee detection
- **50% memory reduction** enabling higher resolution processing
- **Simplified maintenance** through modern framework

## Target Architecture

### Current Architecture (TO BE REPLACED)
```
Frame → YOLOv4 (TensorFlow 2.3.0) → filter_boxes() → DeepSORT → Visualization
```

### Target Architecture (TO BE IMPLEMENTED)
```
Frame → YOLOv10 (PyTorch/Ultralytics) → DetectionAdapter → DeepSORT → Visualization
```

## Model Selection Recommendation

**Primary Choice: YOLOv10**
- NMS-free design reduces latency by 40-60%
- Highest accuracy-to-parameter ratio
- Modern ecosystem with active support
- Excellent for real-time edge deployment

**Fallback Option: YOLOv8**
- Most stable and extensively tested
- Good performance across diverse hardware
- Broader documentation and community support
- Easier migration path

---

## Phase 1: Framework Selection & Planning (1-2 weeks)

### ☐ Task 1.1: Research and Finalize Model Choice
**Estimated Time:** 2-3 days  
**Assignee:** Junior Developer  
**Requirements:**
- [ ] Review YOLOv10 vs YOLOv8 performance benchmarks
- [ ] Test both models on sample Ultimate frisbee footage
- [ ] Evaluate hardware compatibility with your development machine
- [ ] Document decision rationale in `Project/model-selection-report.md`

**Deliverable:** 
- Model selection report with performance comparisons
- Recommended model variant (e.g., YOLOv10n vs YOLOv10s)

### ☐ Task 1.2: Create Development Environment Setup Guide
**Estimated Time:** 1-2 days  
**Assignee:** Junior Developer  
**Requirements:**
- [ ] Document current pixi environment configuration
- [ ] Create new pixi environment for YOLO migration
- [ ] Install Ultralytics framework and dependencies via pixi
- [ ] Test basic YOLO model loading and inference
- [ ] Write environment setup guide in `Project/environment-setup.md`

**Deliverable:** 
- Working pixi environment with modern YOLO
- Step-by-step setup documentation

### ☐ Task 1.3: Create Migration Branch
**Estimated Time:** 1 day  
**Assignee:** Junior Developer  
**Requirements:**
- [ ] Create git branch: `feature/yolo-migration`
- [ ] Tag current stable version: `yolov4-final`
- [ ] Document current performance baseline in `Project/baseline-metrics.md`

**Deliverable:**
- Working branch for migration development
- Baseline performance metrics for comparison

---

## Phase 2: Dependencies & Environment Setup (1 week)

### ☐ Task 2.1: Update Project Dependencies
**Estimated Time:** 2-3 days  
**Assignee:** Junior Developer  
**Requirements:**
- [ ] Update `pyproject.toml` with new dependencies
- [ ] Remove TensorFlow-specific dependencies
- [ ] Add PyTorch and Ultralytics packages
- [ ] Update pixi tasks for new framework
- [ ] Test dependency installation on clean environment

**Code Changes Required:**
```toml
# In pyproject.toml, REPLACE:
[tool.pixi.pypi-dependencies]
tensorflow-cpu = "~=2.3.0"
opencv-python = "~=4.1.0"

# WITH:
[tool.pixi.pypi-dependencies]
torch = ">=2.0.0"
torchvision = ">=0.15.0"
ultralytics = ">=8.0.0"
opencv-python = ">=4.8.0"
```

### ☐ Task 2.2: Update Project Structure
**Estimated Time:** 2 days  
**Assignee:** Junior Developer  
**Requirements:**
- [ ] Create new `yolov10/` directory structure
- [ ] Mark old files for deletion (but keep during migration)
- [ ] Create module skeleton files
- [ ] Update imports across project

**Directory Structure to Create:**
```
yolov10/
├── __init__.py
├── model_loader.py          # YOLOv10 model management
├── config.py              # YOLOv10 specific configurations
├── inference.py            # Inference wrapper
└── utils.py              # YOLOv10 utility functions
```

### ☐ Task 2.3: Update Configuration System
**Estimated Time:** 1-2 days  
**Assignee:** Junior Developer  
**Requirements:**
- [ ] Modify `core/config.py` to support new YOLO versions
- [ ] Add model variant configurations (n/s/m/b/l/x)
- [ ] Create hardware-specific optimization settings
- [ ] Maintain backward compatibility with existing config format

**Code Changes Example:**
```python
# Add to core/config.py:
__C.YOLOV10 = edict()
__C.YOLOV10.MODEL_VARIANT = "yolov10n"  # n, s, m, b, l, x
__C.YOLOV10.CONFIDENCE_THRESHOLD = 0.25
__C.YOLOV10.IOU_THRESHOLD = 0.45
__C.YOLOV10.HARDWARE_TARGET = "auto"  # auto, cpu, gpu, tensorrt
```

---

## Phase 3: Core Integration (2-3 weeks)

### ☐ Task 3.1: Create Detection Adapter Interface
**Estimated Time:** 3-4 days  
**Assignee:** Junior Developer  
**Requirements:**
- [ ] Create `yolov10/detection_adapter.py`
- [ ] Implement format conversion from YOLOv10 to DeepSORT input
- [ ] Maintain exact same interface as `filter_boxes()` function
- [ ] Add comprehensive unit tests
- [ ] Test with sample detection outputs

**Implementation Skeleton:**
```python
# yolov10/detection_adapter.py
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple

class DetectionAdapter:
    """Convert YOLOv10 output to DeepSORT compatible format"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.25):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Detect objects in frame and return DeepSORT-compatible format
        
        Returns:
            bboxes: numpy array of bounding boxes [x, y, w, h]
            scores: numpy array of confidence scores
            classes: numpy array of class indices
            num_objects: number of detected objects
        """
        # TODO: Implement YOLOv10 inference
        # TODO: Convert to DeepSORT format
        pass
    
    def filter_by_class(self, detections, allowed_classes: List[str]) -> np.ndarray:
        """Filter detections to only include specified classes (person, frisbee)"""
        # TODO: Implement class filtering
        pass
```

### ☐ Task 3.2: Replace YOLOv4 Inference Pipeline
**Estimated Time:** 4-5 days  
**Assignee:** Junior Developer  
**Requirements:**
- [ ] Update `object_tracker.py` to use new DetectionAdapter
- [ ] Replace YOLOv4 model loading with YOLOv10
- [ ] Update tensor conversion and preprocessing
- [ ] Maintain existing command-line interface
- [ ] Test with sample videos

**Key Changes in object_tracker.py:**
```python
# REPLACE these lines:
# from core.yolov4 import filter_boxes
# saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
# infer = saved_model_loaded.signatures["serving_default"]

# WITH these lines:
# from yolov10.detection_adapter import DetectionAdapter
# detector = DetectionAdapter(FLAGS.weights, FLAGS.score)
```

### ☐ Task 3.3: Update Model Loading and Saving
**Estimated Time:** 2-3 days  
**Assignee:** Junior Developer  
**Requirements:**
- [ ] Create new `save_model.py` for YOLOv10
- [ ] Implement model export for different formats (ONNX, TensorRT)
- [ ] Add model variant selection (n/s/m/b/l/x)
- [ ] Maintain CLI compatibility
- [ ] Add performance optimization options

**New save_model.py Features:**
```python
# Add to save_model.py:
def save_yolov10_model(model_variant: str, output_path: str, optimize_for: str):
    """
    Save optimized YOLOv10 model for specific deployment target
    
    Args:
        model_variant: yolo10n, yolov10s, yolov10m, yolov10b, yolov10x
        output_path: Path to save optimized model
        optimize_for: 'cpu', 'gpu', 'tensorrt', 'tflite', 'coreml'
    """
    # TODO: Implement model loading and export
    pass
```

---

## Phase 4: Performance Optimization (1-2 weeks)

### ☐ Task 4.1: Implement Hardware-Specific Optimizations
**Estimated Time:** 4-5 days  
**Assignee:** Junior Developer  
**Requirements:**
- [ ] Add TensorRT export and inference support
- [ ] Implement ONNX Runtime for CPU optimization
- [ ] Add CoreML export for Apple Silicon
- [ ] Create TensorFlow Lite export for embedded devices
- [ ] Benchmark each optimization target

**Optimization Implementation:**
```python
# yolov10/optimization.py
class ModelOptimizer:
    
    def optimize_for_tensorrt(self, model, precision: str = "fp16"):
        """Export model for TensorRT inference"""
        pass
    
    def optimize_for_onnx(self, model):
        """Export model for ONNX Runtime optimization"""
        pass
    
    def optimize_for_tflite(self, model, quantize: bool = True):
        """Export model for TensorFlow Lite with quantization"""
        pass
```

### ☐ Task 4.2: Create Performance Benchmarking Suite
**Estimated Time:** 3-4 days  
**Assignee:** Junior Developer  
**Requirements:**
- [ ] Create `benchmark/performance_test.py`
- [ ] Implement FPS measurement across different hardware
- [ ] Add memory usage profiling
- [ ] Create accuracy validation on test dataset
- [ ] Generate comparison reports (YOLOv4 vs YOLOv10)

**Benchmark Suite Structure:**
```python
# benchmark/performance_test.py
class PerformanceBenchmark:
    
    def measure_fps(self, model, video_path: str, duration: int = 60) -> float:
        """Measure average FPS over specified duration"""
        pass
    
    def measure_memory_usage(self, model, resolution: tuple) -> dict:
        """Measure GPU/CPU memory consumption"""
        pass
    
    def validate_accuracy(self, model, test_data_path: str) -> dict:
        """Validate detection accuracy on labeled test data"""
        pass
    
    def generate_report(self, results: dict) -> str:
        """Generate HTML/PDF performance report"""
        pass
```

---

## Phase 5: Testing & Validation (2 weeks)

### ☐ Task 5.1: Comprehensive Unit Testing
**Estimated Time:** 3-4 days  
**Assignee:** Junior Developer  
**Requirements:**
- [ ] Write unit tests for DetectionAdapter
- [ ] Test model loading and inference
- [ ] Validate DeepSORT integration
- [ ] Test configuration management
- [ ] Add tests for error handling and edge cases
- [ ] Achieve >90% code coverage

**Test Structure:**
```
tests/
├── test_detection_adapter.py
├── test_model_loader.py
├── test_optimization.py
├── test_integration.py
└── conftest.py
```

### ☐ Task 5.2: Integration Testing
**Estimated Time:** 4-5 days  
**Assignee:** Junior Developer  
**Requirements:**
- [ ] Test end-to-end pipeline with sample videos
- [ ] Validate output format compatibility with DeepSORT
- [ ] Test command-line interface options
- [ ] Verify export/import functionality
- [ ] Test on different hardware configurations
- [ ] Create integration test report

**Integration Test Checklist:**
```python
# tests/test_integration.py
def test_end_to_end_pipeline():
    """Test complete detection + tracking pipeline"""
    pass

def test_command_line_interface():
    """Test all CLI options and arguments"""
    pass

def test_model_export_formats():
    """Test all export formats work correctly"""
    pass
```

---

## Phase 6: Final Implementation (1 week)

### ☐ Task 6.1: Documentation and Deployment Guide
**Estimated Time:** 2-3 days  
**Assignee:** Junior Developer  
**Requirements:**
- [ ] Update README.md with new features
- [ ] Create migration guide for existing users
- [ ] Document new configuration options
- [ ] Write troubleshooting guide
- [ ] Create performance optimization guide

**Documentation Structure:**
```
docs/
├── migration-guide.md
├── configuration.md
├── performance-tuning.md
├── troubleshooting.md
└── api-reference.md
```

### ☐ Task 6.2: Clean-up and Code Removal
**Estimated Time:** 1-2 days  
**Assignee:** Junior Developer  
**Requirements:**
- [ ] Remove deprecated YOLOv4 files after successful testing
- [ ] Clean up unused imports and dependencies
- [ ] Remove legacy configuration options
- [ ] Update AGENTS.md with new pixi task commands
- [ ] Archive old code in separate branch

---

## Phase 7: Advanced Features (Optional, 2-3 weeks)

### ☐ Task 7.1: Multi-Model Support
**Estimated Time:** 5-6 days  
**Assignee:** Junior Developer  
**Requirements:**
- [ ] Implement dynamic model selection based on conditions
- [ ] Add context-aware model switching
- [ ] Create performance-based model adaptation
- [ ] Test with different lighting and motion conditions
- [ ] Add model selection heuristics

### ☐ Task 7.2: Domain-Specific Fine-Tuning
**Estimated Time:** 8-10 days  
**Assignee:** Junior Developer  
**Requirements:**
- [ ] Collect labeled Ultimate frisbee dataset
- [ ] Set up training pipeline with Ultralytics
- [ ] Fine-tune YOLOv10 on sports-specific data
- [ ] Validate improved performance on frisbee footage
- [ ] Create custom model distribution package

---

## Success Criteria

### Performance Targets
- [ ] **60-120+ FPS** on same hardware (vs current 15-30 FPS)
- [ ] **5-10% mAP improvement** for player/frisbee detection
- [ ] **50% memory reduction** enabling higher resolution processing
- [ ] **<50ms latency** from frame to detection

### Quality Targets
- [ ] **90%+ code coverage** with comprehensive tests
- [ ] **Zero breaking changes** to existing DeepSORT integration
- [ ] **Backward compatibility** with existing configuration files
- [ ] **Complete documentation** with examples and troubleshooting

### Deployment Targets
- [ ] **One-click deployment** across different hardware targets
- [ ] **Automated performance optimization** based on detected hardware
- [ ] **Seamless migration** path for existing users
- [ ] **Robust error handling** and fallback mechanisms

---

## Risk Mitigation

### High-Risk Items
1. **DeepSORT Integration Complexity**
   - **Risk:** Detection format incompatibility breaking tracking
   - **Mitigation:** Comprehensive adapter testing + parallel running mode

2. **Performance Regression**
   - **Risk:** New model performs worse than expected
   - **Mitigation:** Extensive benchmarking + performance validation

3. **Hardware Compatibility**
   - **Risk:** YOLOv10 doesn't work on target hardware
   - **Mitigation:** Multiple export formats + YOLOv8 fallback

### Mitigation Checklist
- [ ] Maintain YOLOv4 branch for 1 month post-migration
- [ ] Implement comprehensive error handling and logging
- [ ] Create automated performance monitoring
- [ ] Provide user feedback collection mechanism

---

## Developer Instructions

### Getting Started
1. **Read this entire plan** before starting any task
2. **Set up development environment** following Task 1.2
3. **Work sequentially through phases** - don't skip ahead
4. **Test each component** before moving to next task
5. **Document progress** in each task's completion section
6. **Ask for help** if stuck on any task for >1 day

### Code Standards
- **Follow existing project style** (see AGENTS.md)
- **Use ruff for formatting and linting** (`pixi run ruff format .`, `pixi run ruff check .`)
- **Use mypy for type checking** (`pixi run mypy .`)
- **Write comprehensive tests** for all new code
- **Add docstrings** to all functions and classes
- **Use type hints** throughout the codebase
- **Test on real hardware** before considering task complete

### Progress Tracking
After completing each task:
1. **Mark checkbox as completed** in this document
2. **Update task status** in Project/progress-tracker.md
3. **Commit changes** with descriptive commit messages
4. **Run tests** to ensure nothing is broken
5. **Document any issues** or deviations from plan

---

## Estimated Timeline

| Phase | Duration | Start | End |
|--------|----------|---------|------|
| Phase 1: Planning | 1-2 weeks | Week 1 | Week 2 |
| Phase 2: Setup | 1 week | Week 2 | Week 3 |
| Phase 3: Integration | 2-3 weeks | Week 3 | Week 6 |
| Phase 4: Optimization | 1-2 weeks | Week 6 | Week 8 |
| Phase 5: Testing | 2 weeks | Week 8 | Week 10 |
| Phase 6: Final Implementation | 1 week | Week 10 | Week 11 |
| Phase 7: Advanced | 2-3 weeks | Week 11 | Week 14 |

**Total Estimated Time:** 6-11 weeks (1.5-2.5 months)

---

## Next Steps

1. **Review and approve this plan** with project stakeholders
2. **Assign Phase 1 tasks** to junior developer
3. **Set up regular check-ins** (weekly progress reviews)
4. **Prepare development environment** and test hardware
5. **Begin Phase 1 implementation**

*This plan is designed to be executed by a junior Python developer working independently, with clear deliverables and success criteria for each task.*