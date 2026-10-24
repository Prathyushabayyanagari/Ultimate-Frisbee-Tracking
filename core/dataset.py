#! /usr/bin/env python
# coding=utf-8

import os
import cv2
import random
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg


class Dataset(object):

    def __init__(self, FLAGS, is_training: bool, dataset_type: str = "converted_coco"):
        """
    Initializes the Dataset class with configuration parameters and loads annotations.

    Parameters:
    - FLAGS (Namespace): Configuration flags containing model details (e.g., tiny YOLO, paths, etc.).
    - is_training (bool): Indicates whether the dataset is for training or testing.
    - dataset_type (str, optional): Specifies the dataset type (default is "converted_coco").

    Attributes:
    - tiny (bool): Determines whether a tiny version of YOLO is used.
    - strides (list), anchors (list), NUM_CLASS (int), XYSCALE (list): Model-specific parameters loaded from FLAGS.
    - dataset_type (str): Stores the type of dataset being used.
    - annot_path (str): Path to annotation files (training or testing based on is_training).
    - input_sizes (int): Input image size.
    - batch_size (int): Number of samples per batch.
    - data_aug (bool): Flag for applying data augmentation.
    - train_input_sizes (int): Training-specific input sizes.
    - classes (dict): Mapping of class IDs to names.
    - num_classes (int): Total number of classes in the dataset.
    - anchor_per_scale (int): Number of anchors per scale.
    - max_bbox_per_scale (int): Maximum number of bounding boxes per scale.
    - annotations (list): List of parsed annotations from the dataset.
    - num_samples (int): Total number of samples in the dataset.
    - num_batchs (int): Total number of batches (computed from samples and batch size).
    - batch_count (int): Counter for tracking current batch during iteration.
    """
        self.tiny = FLAGS.tiny
        self.strides, self.anchors, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        self.dataset_type = dataset_type

        self.annot_path = cfg.TRAIN.ANNOT_PATH if is_training else cfg.TEST.ANNOT_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if is_training else cfg.TEST.INPUT_SIZE
        self.batch_size = cfg.TRAIN.BATCH_SIZE if is_training else cfg.TEST.BATCH_SIZE
        self.data_aug = cfg.TRAIN.DATA_AUG if is_training else cfg.TEST.DATA_AUG

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations()
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def load_annotations(self):
    """
    Loads and parses annotation data from the specified annotation file.

    Depending on the dataset type (`converted_coco` or `yolo`), this method processes
    annotations differently. The annotations include image paths and bounding box details,
    which are used for object detection.

    Returns:
    - annotations (list): A list of annotation strings. Each string represents an image 
      path followed by the bounding box details in the format:
      "x_min,y_min,x_max,y_max,class_id", where:
        - x_min, y_min: Top-left corner coordinates of the bounding box.
        - x_max, y_max: Bottom-right corner coordinates of the bounding box.
        - class_id: The class index of the detected object.

    Notes:
    - For `converted_coco`, lines with valid annotations are directly added to the list.
    - For `yolo`, annotations are constructed by reading bounding box data from
      accompanying `.txt` files, transforming YOLO format to the required format.
    - Shuffles the annotation list to randomize the data order.
    """
        with open(self.annot_path, "r") as f:
            txt = f.readlines()
            if self.dataset_type == "converted_coco":
                annotations = [
                    line.strip() for line in txt if len(line.strip().split()[1:]) != 0
                ]
            elif self.dataset_type == "yolo":
                annotations = []
                for line in txt:
                    image_path = line.strip()
                    root, _ = os.path.splitext(image_path)
                    with open(root + ".txt") as fd:
                        boxes = fd.readlines()
                        string = ""
                        for box in boxes:
                            box = box.strip()
                            box = box.split()
                            class_num = int(box[0])
                            center_x = float(box[1])
                            center_y = float(box[2])
                            half_width = float(box[3]) / 2
                            half_height = float(box[4]) / 2
                            string += " {},{},{},{},{}".format(
                                center_x - half_width,
                                center_y - half_height,
                                center_x + half_width,
                                center_y + half_height,
                                class_num,
                            )
                        annotations.append(image_path + string)

        np.random.shuffle(annotations)
        return annotations

    def __iter__(self):
        return self

    def __next__(self):
    """
    Generates the next batch of data for training or testing.

    This method creates batches of images, labels, and bounding box targets for
    the model. It preprocesses the data for three different scales (small, medium,
    and large) used in YOLOv4.

    Returns:
    - batch_image (np.ndarray): A batch of images with shape 
      (batch_size, input_size, input_size, 3).
    - batch_targets (tuple): A tuple containing targets for three scales:
        - batch_smaller_target: (batch_label_sbbox, batch_sbboxes)
        - batch_medium_target: (batch_label_mbbox, batch_mbboxes)
        - batch_larger_target: (batch_label_lbbox, batch_lbboxes)

      Each scale includes:
      - batch_label_sbbox (np.ndarray): Label tensor for the smaller scale.
      - batch_sbboxes (np.ndarray): Bounding boxes tensor for the smaller scale.
      - The same structure applies to `batch_label_mbbox` / `batch_mbboxes` and 
        `batch_label_lbbox` / `batch_lbboxes` for medium and large scales.

    Raises:
    - StopIteration: If all batches in the current epoch have been processed. 
      The annotations are shuffled, and the batch counter is reset for the next epoch.

    Notes:
    - Creates zero-initialized arrays for images, labels, and bounding boxes and 
      populates them by processing annotations via `parse_annotation()` and 
      `preprocess_true_boxes()`.
    - Works on CPU (`/cpu:0`) to avoid GPU memory issues.
    - Resets the batch counter when all data is processed and shuffles annotations for 
      the next epoch.
    """
        with tf.device("/cpu:0"):
            # self.train_input_size = random.choice(self.train_input_sizes)
            self.train_input_size = cfg.TRAIN.INPUT_SIZE
            self.train_output_sizes = self.train_input_size // self.strides

            batch_image = np.zeros(
                (
                    self.batch_size,
                    self.train_input_size,
                    self.train_input_size,
                    3,
                ),
                dtype=np.float32,
            )

            batch_label_sbbox = np.zeros(
                (
                    self.batch_size,
                    self.train_output_sizes[0],
                    self.train_output_sizes[0],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                ),
                dtype=np.float32,
            )
            batch_label_mbbox = np.zeros(
                (
                    self.batch_size,
                    self.train_output_sizes[1],
                    self.train_output_sizes[1],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                ),
                dtype=np.float32,
            )
            batch_label_lbbox = np.zeros(
                (
                    self.batch_size,
                    self.train_output_sizes[2],
                    self.train_output_sizes[2],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                ),
                dtype=np.float32,
            )

            batch_sbboxes = np.zeros(
                (self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32
            )
            batch_mbboxes = np.zeros(
                (self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32
            )
            batch_lbboxes = np.zeros(
                (self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32
            )

            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:
                        index -= self.num_samples
                    annotation = self.annotations[index]
                    image, bboxes = self.parse_annotation(annotation)
                    (
                        label_sbbox,
                        label_mbbox,
                        label_lbbox,
                        sbboxes,
                        mbboxes,
                        lbboxes,
                    ) = self.preprocess_true_boxes(bboxes)

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target = batch_label_mbbox, batch_mbboxes
                batch_larger_target = batch_label_lbbox, batch_lbboxes

                return (
                    batch_image,
                    (
                        batch_smaller_target,
                        batch_medium_target,
                        batch_larger_target,
                    ),
                )
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def random_horizontal_flip(self, image, bboxes):
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

        return image, bboxes

    def random_crop(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate(
                [
                    np.min(bboxes[:, 0:2], axis=0),
                    np.max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate(
                [
                    np.min(bboxes[:, 0:2], axis=0),
                    np.max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def parse_annotation(self, annotation):
    """
    Parses a single annotation entry to load and preprocess the corresponding image 
    and bounding boxes.

    Parameters:
    - annotation (str): A string containing the image path and bounding box data. 
      The format depends on `dataset_type`:
        - For "converted_coco": "image_path x_min,y_min,x_max,y_max,class_id ..."
        - For "yolo": "image_path x_center,y_center,width,height,class_id ..."

    Returns:
    - image (np.ndarray): The preprocessed image as a NumPy array with dimensions 
      [self.train_input_size, self.train_input_size, 3].
    - bboxes (np.ndarray): The preprocessed bounding boxes as a NumPy array. Each bounding 
      box is represented in the format [x_min, y_min, x_max, y_max, class_id].

    Raises:
    - KeyError: If the specified image file does not exist.

    Notes:
    - Bounding Box Transformation:
        - For "converted_coco", bounding boxes are directly read as integers.
        - For "yolo", bounding boxes are initially in YOLO format and are transformed to
          corner coordinates using the image dimensions.
    - Data Augmentation:
        - If `self.data_aug` is True, applies random horizontal flip, cropping, and 
          translation to the image and bounding boxes.
    - Image Preprocessing:
        - Converts the image from BGR (OpenCV default) to RGB.
        - Resizes the image and adjusts bounding boxes to match the target input size 
          using `utils.image_preprocess`.

    """
        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        image = cv2.imread(image_path)
        if self.dataset_type == "converted_coco":
            bboxes = np.array([list(map(int, box.split(","))) for box in line[1:]])
        elif self.dataset_type == "yolo":
            height, width, _ = image.shape
            bboxes = np.array([list(map(float, box.split(","))) for box in line[1:]])
            bboxes = bboxes * np.array([width, height, width, height, 1])
            bboxes = bboxes.astype(np.int64)

        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, bboxes = utils.image_preprocess(
            np.copy(image),
            [self.train_input_size, self.train_input_size],
            np.copy(bboxes),
        )
        return image, bboxes

    def preprocess_true_boxes(self, bboxes):
    """
    Prepares ground-truth bounding boxes for use in the YOLOv4 model.

    This method processes the input bounding boxes to create label tensors for
    three scales (small, medium, and large) and assigns each bounding box to the
    most appropriate anchor based on Intersection Over Union (IoU) values.

    Parameters:
    - bboxes (np.ndarray): A NumPy array containing ground-truth bounding boxes, where
      each box is represented as [x_min, y_min, x_max, y_max, class_id].

    Returns:
    - label_sbbox (np.ndarray): Label tensor for the small-scale feature map.
    - label_mbbox (np.ndarray): Label tensor for the medium-scale feature map.
    - label_lbbox (np.ndarray): Label tensor for the large-scale feature map.
    - sbboxes (np.ndarray): Bounding boxes assigned to the small-scale feature map.
    - mbboxes (np.ndarray): Bounding boxes assigned to the medium-scale feature map.
    - lbboxes (np.ndarray): Bounding boxes assigned to the large-scale feature map.

    Notes:
    - Each `label` tensor has the shape 
      [feature_map_size, feature_map_size, anchor_per_scale, 5 + num_classes],
      where the last dimension includes:
        - [x_center, y_center, width, height, objectness_score, class_onehot_vector].
    - Smooth Labeling:
        - Applies label smoothing to reduce overconfidence in class probabilities.
    - IoU Assignment:
        - Assigns bounding boxes to anchors based on IoU. If no anchor exceeds the IoU
          threshold (0.3), the best anchor is selected.
    - Bounding Box Transformation:
        - Converts bounding boxes from corner coordinates [x_min, y_min, x_max, y_max]
          to center coordinates [x_center, y_center, width, height].

    """
        label = [
            np.zeros(
                (
                    self.train_output_sizes[i],
                    self.train_output_sizes[i],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                )
            )
            for i in range(3)
        ]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate(
                [
                    (bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                    bbox_coor[2:] - bbox_coor[:2],
                ],
                axis=-1,
            )
            bbox_xywh_scaled = (
                1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
            )

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = (
                    np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                )
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = utils.bbox_iou(
                    bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh
                )
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(
                    np.int32
                )

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs
