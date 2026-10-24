#! /usr/bin/env python
from easydict import EasyDict as edict

# Create a configuration dictionary using EasyDict for a flexible and hierarchical structure
__C = edict()
cfg = __C  # Expose the configuration dictionary via 'cfg'

# YOLO-specific options
__C.YOLO = edict()

# Path to the file containing class names.
__C.YOLO.CLASSES = "./data/classes/coco.names"
# Anchor box dimensions for different model versions
__C.YOLO.ANCHORS = [
    12,
    16,
    19,
    36,
    40,
    28,
    36,
    75,
    76,
    55,
    72,
    146,
    142,
    110,
    192,
    243,
    459,
    401,
]
__C.YOLO.ANCHORS_V3 = [
    10,
    13,
    16,
    30,
    33,
    23,
    30,
    61,
    62,
    45,
    59,
    119,
    116,
    90,
    156,
    198,
    373,
    326,
]
__C.YOLO.ANCHORS_TINY = [23, 27, 37, 58, 81, 82, 81, 82, 135, 169, 344, 319]
# Stride values for each detection scale (grid cell size multipliers)
__C.YOLO.STRIDES = [8, 16, 32]
__C.YOLO.STRIDES_TINY = [16, 32]
# Scaling factors for bounding box center adjustments
__C.YOLO.XYSCALE = [1.2, 1.1, 1.05]
__C.YOLO.XYSCALE_TINY = [1.05, 1.05]
# Number of anchor boxes per detection scale
__C.YOLO.ANCHOR_PER_SCALE = 3
# IoU loss threshold for identifying foreground vs background boxes
__C.YOLO.IOU_LOSS_THRESH = 0.5

# Training-specific options
__C.TRAIN = edict()

# Path to the training annotation file.
__C.TRAIN.ANNOT_PATH = "./data/dataset/val2017.txt"
# Number of images per training batch
__C.TRAIN.BATCH_SIZE = 2
# Input image size for training.
__C.TRAIN.INPUT_SIZE = 416
# Enable or disable data augmentation during training
__C.TRAIN.DATA_AUG = True
# Initial learning rate
__C.TRAIN.LR_INIT = 1e-3
# Final learning rate after decay
__C.TRAIN.LR_END = 1e-6
# Number of warmup epochs (gradual learning rate increase)
__C.TRAIN.WARMUP_EPOCHS = 2
# Number of epochs for the first stage of training (e.g., freezing backbone layers)
__C.TRAIN.FISRT_STAGE_EPOCHS = 20
# Number of epochs for the second stage of training (e.g., training entire network)
__C.TRAIN.SECOND_STAGE_EPOCHS = 30

# Testing-specific options
__C.TEST = edict()

# Path to the testing annotation file.
__C.TEST.ANNOT_PATH = "./data/dataset/val2017.txt"
# Number of images per testing batch
__C.TEST.BATCH_SIZE = 2
# Input image size for testing
__C.TEST.INPUT_SIZE = 416
# Enable or disable data augmentation during testing
__C.TEST.DATA_AUG = False
# Path to save detected images after inference
__C.TEST.DECTECTED_IMAGE_PATH = "./data/detection/"
# Confidence score threshold for filtering detections
__C.TEST.SCORE_THRESHOLD = 0.25
# IoU threshold for Non-Maximum Suppression (NMS) during testing
__C.TEST.IOU_THRESHOLD = 0.5

# YOLOv10-specific options
__C.YOLOV10 = edict()

# Model variant selection (n, s, m, b, l, x)
__C.YOLOV10.MODEL_VARIANT = "yolov10s"
# Detection confidence threshold
__C.YOLOV10.CONFIDENCE_THRESHOLD = 0.25
# IoU threshold for detection filtering
__C.YOLOV10.IOU_THRESHOLD = 0.45
# Hardware optimization target (auto, cpu, gpu, tensorrt)
__C.YOLOV10.HARDWARE_TARGET = "auto"
# Input image size
__C.YOLOV10.INPUT_SIZE = 640
# Enable half precision (FP16) for GPU
__C.YOLOV10.HALF_PRECISION = True
# Maximum detections per image
__C.YOLOV10.MAX_DETECTIONS = 300
# Target classes for Ultimate frisbee (person, sports ball)
__C.YOLOV10.TARGET_CLASSES = [0, 37]
# Export format for deployment (onnx, torchscript, coreml, tflite)
__C.YOLOV10.EXPORT_FORMAT = "onnx"
# DeepSORT compatibility mode
__C.YOLOV10.DEEPSORT_COMPATIBLE = True
