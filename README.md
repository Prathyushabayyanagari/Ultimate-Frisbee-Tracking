# UltimatePlayerTracker

Tracking of players on the field for Ultimate. 
This project is based on a more general workflow of object tracking implemented with YOLOv10, DeepSort, and PyTorch. YOLOv10 is an algorithm that uses deep convolutional neural networks for object detections. These detections are then fed to into DeepSORT (Simple Online and Realtime Tracking with a Deep Association Metric) in order to create a highly accurate object tracker.

## Work in progress
This project is currently at a proof of consept stage. The algorithm is more tuned for tracking pedestrians and would benefit from finetuning to the players actively making fakes. Also, we could achieve a much better performance of staying on playes, if we leverage the fact thast there are always sa set number of players on field for any given point.

Some first steps for the project could be
  - Obtaining automatic video processing by recognising where a point ends and another starts.
  - Automated labelling for points; offence, defence, break, etc.
  - Statistics that also contain information on the movements of off-disc players -- most current apps for statistics tracking focus on the movement of the  disc, and completion raters of throwers.

## Getting Started
First, install project dependencies using Pixi (recommended) or the legacy methods.

### Pixi (Recommended - Modern)

```bash
# Install Pixi package manager (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Install all dependencies
pixi install

# Activate the environment
pixi shell
```

### Conda (Legacy)

```bash
# For YOLOv4 legacy support
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# For YOLOv4 GPU legacy support
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

### Pip (Legacy)
```bash
# For YOLOv4 legacy CPU support
pip install -r requirements.txt

# For YOLOv4 legacy GPU support
pip install -r requirements-gpu.txt

# For YOLOv10 modern support
pip install torch torchvision ultralytics opencv-python
```
### Nvidia Driver (For GPU, if you are not using Conda Environment and haven't set up CUDA yet)
Make sure to use CUDA Toolkit version 10.1 as it is the proper version for the TensorFlow version used in this repository.
https://developer.nvidia.com/cuda-10.1-download-archive-update2

## YOLOv10 Model Setup

Our object tracker uses YOLOv10 for making the player (and in the future disc) detections. YOLOv10 provides state-of-the-art performance with multiple model variants optimized for different use cases.

### Available YOLOv10 Variants
- **YOLOv10n**: Nano - Fastest, suitable for real-time applications
- **YOLOv10s**: Small - Balanced speed and accuracy (default)
- **YOLOv10m**: Medium - Higher accuracy
- **YOLOv10b**: Base - Even better accuracy
- **YOLOv10l**: Large - High accuracy
- **YOLOv10x**: Extra Large - Best accuracy

### Automatic Model Download
The first time you run the tracker with YOLOv10, the model will be automatically downloaded. No manual download required.

### Manual Model Download (Optional)
If you prefer to download models manually:
```bash
# Download YOLOv10s (recommended)
wget https://github.com/THU-MIG/yolov10/releases/download/v1.0/yolov10s.pt

# Place in project root or data/ folder
```

### Legacy YOLOv4 Support
YOLOv4 support is maintained for comparison and legacy purposes. See the "Running the Tracker with YOLOv4 (Legacy)" section below if needed.

## Running the Tracker with YOLOv10

### Quick Start
```bash
# Run with default YOLOv10s model
pixi run python object_tracker.py --video ./data/video/demo.mp4 --output ./outputs/demo.avi --model yolov10

# Run with webcam (set video flag to 0)
pixi run python object_tracker.py --video 0 --output ./outputs/webcam.avi --model yolov10
```

### Webcam website (Windows)

Start the local website from the project folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe .\app.py
```

The app opens `http://127.0.0.1:8765` in your browser. Select a camera and
inference size, click **Start tracking**, and allow webcam access when prompted.
The annotated live preview and controls are shown together in the website.
Click **Stop session** to release the camera. Press `Ctrl+C` in the PowerShell
terminal to stop the website server.

The live detection analysis table shows confirmed player track IDs, latest
person detection confidence and bounding-box coordinates. It also shows raw
model `frisbee` and `sports ball` candidates, plus a white oval/circle shape
heuristic labeled `frisbee-like`. The shape heuristic is not a trained
classifier and can match white bottles or other objects; it is not proof of a
frisbee. Generic sports-ball predictions are not labeled as confirmed
frisbees.
The webcam interface defaults to input size `416` and person confidence `0.40`
for a faster, more sensitive preview. Choose `0.25` to show more detections,
or `0.55` to reduce weak predictions. Tracks must appear in two consecutive
detections before being displayed. Lower confidence can introduce false
positives; higher confidence can miss distant players.

The server binds only to `127.0.0.1`; camera frames are sent to the local
Python process and are not uploaded to a remote service. The website uses
browser webcam access, which is supported on localhost in current browsers.

Inference size `960` is the default to retain detail for small objects; choose
`416` or `640` for a faster preview. Higher sizes cost more processing time,
and neither larger input nor `sports ball` candidates guarantees that the COCO
model will identify an Ultimate Frisbee disc. CPU processing may be slower than
the webcam's frame rate; the website submits one frame at a time and displays
inference FPS. Use `auto` only when a supported accelerator is available.

### Advanced Usage
```bash
# Use specific YOLOv10 variant
pixi run python object_tracker.py --video ./data/video/demo.mp4 --model yolov10n --output ./outputs/fast.avi

# GPU acceleration
pixi run python object_tracker.py --video ./data/video/demo.mp4 --model yolov10m --device cuda

# Custom confidence and IOU thresholds
pixi run python object_tracker.py --video ./data/video/demo.mp4 --model yolov10s --confidence 0.3 --iou 0.5
```

### Model Management
```bash
# Download and export specific variant
pixi run python save_model.py --model yolov10s --export_format onnx

# Download all variants
pixi run python save_model.py --download_all

# Benchmark performance
pixi run python save_model.py --benchmark
```

## Running the Tracker with YOLOv4 (Legacy)
Legacy YOLOv4 support is maintained for comparison purposes.
```bash
# Convert darknet weights to tensorflow model
pixi run python save_model.py --model yolov4

# Run yolov4 deep sort object tracker on video
pixi run python object_tracker.py --video ./data/video/demo.mp4 --output ./outputs/demo.avi --model yolov4
```
The ``--output`` flag saves the resulting video of the object tracker to the specified location.

## Performance Comparison

### YOLOv10 Variants Performance
- **YOLOv10n**: 60-120 FPS (fastest)
- **YOLOv10s**: 40-80 FPS (balanced)
- **YOLOv10m**: 25-50 FPS (accurate)
- **YOLOv10l**: 15-30 FPS (very accurate)
- **YOLOv10x**: 10-20 FPS (most accurate)

### YOLOv10 vs YOLOv4
- **Speed**: 3-4x faster inference
- **Accuracy**: 5-10% better mAP
- **Memory**: 50% reduction
- **Framework**: Modern PyTorch vs legacy TensorFlow

### Command Line Args Reference

```bash
save_model.py:
  --model: yolov10n, yolov10s, yolov10m, yolov10b, yolov10l, yolov10x (default: yolov10s)
  --export_format: onnx, torchscript, coreml, tflite (default: onnx)
  --download_all: download all model variants
  --benchmark: run performance benchmarking
     
object_tracker.py:
  --video: path to input video (use 0 for webcam)
    (default: './data/video/demo.mp4')
  --output: path to output video
    (default: None)
  --model: yolov10n, yolov10s, yolov10m, yolov10b, yolov10l, yolov10x, yolov4 (default: yolov10)
  --device: cpu, cuda, mps (auto-detected, default: auto)
  --confidence: confidence threshold
    (default: 0.25)
  --iou: iou threshold
    (default: 0.45)
  --output_df: path to output file containing player locations
    (default: None)
  --dont_show: dont show video output
    (default: False)
  --info: print detailed info about tracked objects
    (default: False)

Legacy YOLOv4 args:
  --weights: path to weights file
    (default: './checkpoints/yolov4-416')
  --framework: tf, trt, tflite (default: tf)
  --size: resize images to (default: 416)
  --tiny: use yolov4-tiny (default: false)
```

## Resulting Video
The resulting video will be saved to where you set the ``--output`` command line flag path to. You can also change the format of the video saved with the ``--output_format`` flag, by default it is set to AVI codec which is XVID.

## Command Line Args Reference

```bash
save_model.py:
  --weights: path to weights file
    (default: './data/yolov4.weights')
  --output: path to output
    (default: './checkpoints/yolov4-416')
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'False')
  --input_size: define input size of export model
    (default: 416)
  --framework: what framework to use (tf, trt, tflite)
    (default: tf)
  --model: yolov3 or yolov4
    (default: yolov4)
    
 object_tracker.py:
  --video: path to input video (use 0 for webcam)
    (default: './data/video/test.mp4')
  --output: path to output video (remember to set right codec for given format. e.g. XVID for .avi)
    (default: None)
  --output_format: codec used in VideoWriter when saving video to file
    (default: 'XVID)
  --output_df: path to output file containing the player locations from confirmed tracks in the video.
    (default: None)
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './checkpoints/yolov4-416')
  --framework: what framework to use (tf, trt, tflite)
    (default: tf)
  --model: yolov3 or yolov4
    (default: yolov4)
  --size: resize images to
    (default: 416)
  --limits: only track detections inside the given limits. Still work in progress.
    (default: False)
  --iou: iou threshold
    (default: 0.45)
  --score: confidence threshold
    (default: 0.50)
  --dont_show: dont show video output
    (default: False)
  --info: print detailed info about tracked objects
    (default: False)
```

## Acknowledgements 

I want to thank the following open-source projects and contributors:

**Original Foundation:**
* [yolov4-deepsort](https://github.com/theAIGuysCode/yolov4-deepsort) - TheAiGuy for the original YOLOv4 + DeepSort implementation
* [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite) - hunglc007 for TensorFlow YOLOv4 implementation
* [Deep SORT Repository](https://github.com/nwojke/deep_sort) - nwojke for the original DeepSORT algorithm

**YOLOv10 Integration:**
* [YOLOv10](https://github.com/THU-MIG/yolov10) - THU-MIG lab for the state-of-the-art YOLOv10 implementation
* [Ultralytics](https://github.com/ultralytics/ultralytics) - For the YOLO ecosystem and PyTorch implementation

This project combines these excellent works to create a specialized Ultimate frisbee player tracking system.

## Frisbee detection diagnostics and custom training

The checked-in `checkpoints/yolov10s.pt` checkpoint uses COCO classes. Its
verified class IDs are `0=person`, `29=frisbee`, and `32=sports ball`. A COCO
`sports ball` prediction is only a candidate; it is not proof that the object is
an Ultimate Frisbee disc. Small, distant, blurred, partly hidden, or unusually
oriented discs may not be detected; lighting, exposure, motion blur, and
oblique camera angles can further reduce recall. CPU inference is also
substantially slower than real time, especially when using overlapping crops.

Run the tracker from Windows CMD with the project's Python environment:

```cmd
.venv\Scripts\python.exe object_tracker.py --video data\video\demo.mp4 --output outputs\frisbee_test.avi --model yolov10s --device cpu --dont_show --output_format MJPG
```

The tracker creates the output directory when needed and draws raw predictions
named `frisbee` in orange before DeepSORT confirms tracks. `sports ball`
predictions are not drawn as frisbee boxes or sent to player tracking. The
player confidence threshold remains separate from the lower disc-candidate
threshold. A `frisbee` class prediction is still only a model prediction, not
proof that the object is a real disc.

To also save confirmed player locations as CSV, pass `--output_df` a path
prefix (the tracker appends `.csv`), for example
`--output_df outputs\frisbee_test_tracks`.

Compare full-frame inference with overlapping high-resolution tiles on sampled
frames. Tile results are translated into full-frame coordinates and
same-class duplicate boxes are suppressed. The console reports all frisbee and
sports-ball candidates for diagnosis, while the annotated images draw only
predictions named `frisbee`:

```cmd
.venv\Scripts\python.exe test_frisbee_crops.py --video data\video\demo.mp4 --weights checkpoints\yolov10s.pt --timestamps 5 15 30 --output-dir outputs\frisbee_diagnostics
```

Annotated diagnostic images and each candidate's class, confidence, full-frame
coordinates, and source are written/reported. This diagnostic can be slow on
CPU because it performs multiple inferences per sampled frame.

### Train on labeled frames from game videos

When COCO predictions do not reliably identify the disc, collect and annotate
frames from the relevant camera angles. The scripts below use the local
`checkpoints\yolov10s.pt` weights; they do not download a model or start
training automatically.

1. Extract source frames (do not use the already annotated diagnostic images):

   ```cmd
   .venv\Scripts\python.exe tools\extract_frisbee_frames.py data\video\demo.mp4 --interval-seconds 0.5 --output-dir data\frisbee_training\unlabeled
   ```

2. Prepare a COCO-compatible dataset layout and class-name file:

   ```cmd
   .venv\Scripts\python.exe tools\train_frisbee_detector.py --prepare-only
   ```

3. Manually move selected source frames into `images\train` and `images\val`,
   then annotate them in YOLO format under the matching `labels` directories.
   Use class `0` for `person` and class `29` for `frisbee` to preserve the
   checkpoint's COCO class IDs. Include labeled empty/negative frames, and do
   not label basketballs or uncertain objects as frisbees. Split by game/video
   rather than adjacent frames so near-identical images do not leak between
   training and validation.

4. Train only after every image has a corresponding YOLO label file:

   ```cmd
   .venv\Scripts\python.exe tools\train_frisbee_detector.py --epochs 100 --imgsz 960 --batch 2 --device cpu
   ```

   The best weights are saved under
   `outputs\frisbee_training\yolov10s-frisbee\weights\best.pt`. Validate the
   model on held-out, manually reviewed frames before using it for tracking.
