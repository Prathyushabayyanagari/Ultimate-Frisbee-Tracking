#! /usr/bin/env python

import numpy as np
import tensorflow as tf

import core.backbone as backbone
import core.common as common
import core.utils as utils

# NUM_CLASS       = len(utils.read_class_names(cfg.YOLO.CLASSES))
# STRIDES         = np.array(cfg.YOLO.STRIDES)
# IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
# XYSCALE = cfg.YOLO.XYSCALE
# ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)


def YOLO(input_layer, NUM_CLASS, model="yolov4", is_tiny=False):
    """
    Creates and returns a YOLO model based on the specified type and size.

    Parameters:
    - input_layer (tf.Tensor): The input tensor for the YOLO model.
    - NUM_CLASS (int): The number of classes in the dataset.
    - model (str, optional): The YOLO version to use, either 'yolov4' (default) or 'yolov3'.
    - is_tiny (bool, optional): If True, creates a Tiny version of the YOLO model (default: False).

    Returns:
    - tf.keras.Model: The specified YOLO model, either 'yolov4', 'yolov4_tiny', 'yolov3', or 'yolov3_tiny'.
    """
    if is_tiny:
        if model == "yolov4":
            return YOLOv4_tiny(input_layer, NUM_CLASS)
        elif model == "yolov3":
            return YOLOv3_tiny(input_layer, NUM_CLASS)
    else:
        if model == "yolov4":
            return YOLOv4(input_layer, NUM_CLASS)
        elif model == "yolov3":
            return YOLOv3(input_layer, NUM_CLASS)


def YOLOv3(input_layer, NUM_CLASS):
    route_1, route_2, conv = backbone.darknet53(input_layer)

    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))

    conv_lobj_branch = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(
        conv_lobj_branch, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False
    )

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.upsample(conv)

    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 768, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    conv_mobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(
        conv_mobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False
    )

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)

    conv = common.convolutional(conv, (1, 1, 384, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    conv_sobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(
        conv_sobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False
    )

    return [conv_sbbox, conv_mbbox, conv_lbbox]


def YOLOv4(input_layer, NUM_CLASS):
    route_1, route_2, conv = backbone.cspdarknet53(input_layer)

    route = conv
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.upsample(conv)
    route_2 = common.convolutional(route_2, (1, 1, 512, 256))
    conv = tf.concat([route_2, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    route_1 = common.convolutional(route_1, (1, 1, 256, 128))
    conv = tf.concat([route_1, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    route_1 = conv
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(
        conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False
    )

    conv = common.convolutional(route_1, (3, 3, 128, 256), downsample=True)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(
        conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False
    )

    conv = common.convolutional(route_2, (3, 3, 256, 512), downsample=True)
    conv = tf.concat([conv, route], axis=-1)

    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))

    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(
        conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False
    )

    return [conv_sbbox, conv_mbbox, conv_lbbox]


def YOLOv4_tiny(input_layer, NUM_CLASS):
    route_1, conv = backbone.cspdarknet53_tiny(input_layer)

    conv = common.convolutional(conv, (1, 1, 512, 256))

    conv_lobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_lbbox = common.convolutional(
        conv_lobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False
    )

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)

    conv_mobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_mbbox = common.convolutional(
        conv_mobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False
    )

    return [conv_mbbox, conv_lbbox]


def YOLOv3_tiny(input_layer, NUM_CLASS):
    route_1, conv = backbone.darknet53_tiny(input_layer)

    conv = common.convolutional(conv, (1, 1, 1024, 256))

    conv_lobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_lbbox = common.convolutional(
        conv_lobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False
    )

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)

    conv_mobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_mbbox = common.convolutional(
        conv_mobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False
    )

    return [conv_mbbox, conv_lbbox]


def decode(
    conv_output,
    output_size,
    NUM_CLASS,
    STRIDES,
    ANCHORS,
    i,
    XYSCALE=[1, 1, 1],
    FRAMEWORK="tf",
):
    """
    Decodes YOLO model outputs into bounding boxes and class probabilities.

    Parameters:
    - conv_output (tf.Tensor): The raw output tensor from the YOLO model's convolutional layer.
    - output_size (int): The spatial size of the output feature map.
    - NUM_CLASS (int): The number of classes in the dataset.
    - STRIDES (list): A list of stride values corresponding to feature map scales.
    - ANCHORS (list): Anchor box dimensions used for bounding box predictions.
    - i (int): The index of the current feature map (scale).
    - XYSCALE (list, optional): Scaling factors for bounding box adjustments (default: [1, 1, 1]).
    - FRAMEWORK (str, optional): The framework to use for decoding ('tf', 'trt', or 'tflite', default: 'tf').

    Returns:
    - Decoded output from the specified framework's decoding function. This typically includes
      bounding box coordinates, objectness scores, and class probabilities.

    Notes:
    - `decode_trt`: Used for TensorRT optimization.
    - `decode_tflite`: Used for TensorFlow Lite models.
    - `decode_tf`: The default TensorFlow implementation.
    """
    if FRAMEWORK == "trt":
        return decode_trt(
            conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE
        )
    elif FRAMEWORK == "tflite":
        return decode_tflite(
            conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE
        )
    else:
        return decode_tf(
            conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE
        )


def decode_train(
    conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]
):
    """
    Decodes raw YOLO model output for use during training, generating bounding box predictions
    and associated probabilities.

    Parameters:
    - conv_output (tf.Tensor): Raw output tensor from a YOLO convolutional layer during training.
    - output_size (int): The spatial size of the output feature map (e.g., 13, 26, or 52 for YOLOv4).
    - NUM_CLASS (int): The number of object classes in the dataset.
    - STRIDES (list): List of stride values corresponding to feature map scales.
    - ANCHORS (list): Anchor box dimensions for each scale.
    - i (int, optional): The index of the current feature map scale (default: 0).
    - XYSCALE (list, optional): Scaling factors for bounding box center adjustments (default: [1, 1, 1]).

    Returns:
    - tf.Tensor: A tensor containing concatenated predictions with shape
      (batch_size, num_boxes, 5 + NUM_CLASS). This includes:
        - Bounding box coordinates [x_center, y_center, width, height].
        - Object confidence scores.
        - Class probabilities.
    """
    conv_output = tf.reshape(
        conv_output,
        (tf.shape(conv_output)[0], output_size, output_size, 3, 5 + NUM_CLASS),
    )

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(
        conv_output, (2, 2, 1, NUM_CLASS), axis=-1
    )

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(
        tf.expand_dims(xy_grid, axis=0), [tf.shape(conv_output)[0], 1, 1, 3, 1]
    )

    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (
        (tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid
    ) * STRIDES[i]
    pred_wh = tf.exp(conv_raw_dwdh) * ANCHORS[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def decode_tf(
    conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]
):
    """
    Decodes raw YOLO model output into bounding box coordinates and class probabilities (TensorFlow implementation).

    Parameters:
    - conv_output (tf.Tensor): Raw output tensor from a YOLO convolutional layer.
    - output_size (int): The spatial size of the output feature map (e.g., 13, 26, or 52 for YOLOv4).
    - NUM_CLASS (int): The number of object classes in the dataset.
    - STRIDES (list): List of strides for each scale, corresponding to feature map sizes.
    - ANCHORS (list): Anchor box dimensions for each scale.
    - i (int, optional): The index of the current feature map scale (default: 0).
    - XYSCALE (list, optional): Scaling factors for adjusting bounding box center predictions (default: [1, 1, 1]).

    Returns:
    - pred_xywh (tf.Tensor): Bounding box coordinates with shape (batch_size, num_boxes, 4),
      where each box is represented as [x_center, y_center, width, height].
    - pred_prob (tf.Tensor): Class probabilities with shape (batch_size, num_boxes, NUM_CLASS).
    """
    batch_size = tf.shape(conv_output)[0]
    conv_output = tf.reshape(
        conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS)
    )

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(
        conv_output, (2, 2, 1, NUM_CLASS), axis=-1
    )

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])

    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (
        (tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid
    ) * STRIDES[i]
    pred_wh = tf.exp(conv_raw_dwdh) * ANCHORS[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    pred_prob = pred_conf * pred_prob
    pred_prob = tf.reshape(pred_prob, (batch_size, -1, NUM_CLASS))
    pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))

    return pred_xywh, pred_prob


def decode_tflite(
    conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]
):
    """
    Decodes raw YOLO model output into bounding box coordinates and class probabilities.

    Parameters:
    - conv_output (tf.Tensor): The raw output tensor from a YOLO layer, optimized for TensorFlow Lite.
    - output_size (int): The spatial size of the output feature map (e.g., 13, 26, or 52 for YOLOv4).
    - NUM_CLASS (int): The number of object classes in the dataset.
    - STRIDES (list): List of stride values corresponding to feature map scales.
    - ANCHORS (list): Anchor box dimensions for each scale.
    - i (int, optional): The index of the current feature map scale (default: 0).
    - XYSCALE (list, optional): Scaling factors for bounding box center adjustments (default: [1, 1, 1]).

    Returns:
    - pred_xywh (tf.Tensor): Bounding box coordinates with shape (1, num_boxes, 4), where each box
      is represented as [x_center, y_center, width, height].
    - pred_prob (tf.Tensor): Class probabilities with shape (1, num_boxes, NUM_CLASS).
    """
    (
        conv_raw_dxdy_0,
        conv_raw_dwdh_0,
        conv_raw_score_0,
        conv_raw_dxdy_1,
        conv_raw_dwdh_1,
        conv_raw_score_1,
        conv_raw_dxdy_2,
        conv_raw_dwdh_2,
        conv_raw_score_2,
    ) = tf.split(
        conv_output,
        (2, 2, 1 + NUM_CLASS, 2, 2, 1 + NUM_CLASS, 2, 2, 1 + NUM_CLASS),
        axis=-1,
    )

    conv_raw_score = [conv_raw_score_0, conv_raw_score_1, conv_raw_score_2]
    for idx, score in enumerate(conv_raw_score):
        score = tf.sigmoid(score)
        score = score[:, :, :, 0:1] * score[:, :, :, 1:]
        conv_raw_score[idx] = tf.reshape(score, (1, -1, NUM_CLASS))
    pred_prob = tf.concat(conv_raw_score, axis=1)

    conv_raw_dwdh = [conv_raw_dwdh_0, conv_raw_dwdh_1, conv_raw_dwdh_2]
    for idx, dwdh in enumerate(conv_raw_dwdh):
        dwdh = tf.exp(dwdh) * ANCHORS[i][idx]
        conv_raw_dwdh[idx] = tf.reshape(dwdh, (1, -1, 2))
    pred_wh = tf.concat(conv_raw_dwdh, axis=1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.stack(xy_grid, axis=-1)  # [gx, gy, 2]
    xy_grid = tf.expand_dims(xy_grid, axis=0)
    xy_grid = tf.cast(xy_grid, tf.float32)

    conv_raw_dxdy = [conv_raw_dxdy_0, conv_raw_dxdy_1, conv_raw_dxdy_2]
    for idx, dxdy in enumerate(conv_raw_dxdy):
        dxdy = (
            (tf.sigmoid(dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid
        ) * STRIDES[i]
        conv_raw_dxdy[idx] = tf.reshape(dxdy, (1, -1, 2))
    pred_xy = tf.concat(conv_raw_dxdy, axis=1)
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    return pred_xywh, pred_prob


def decode_trt(
    conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]
):
    """
    Decodes raw YOLO model output into bounding box coordinates and class probabilities.

    Parameters:
    - conv_output (tf.Tensor): Raw output tensor from a YOLO convolutional layer, optimized for TensorRT.
    - output_size (int): The spatial size of the output feature map (e.g., 13, 26, or 52 for YOLOv4).
    - NUM_CLASS (int): The number of object classes in the dataset.
    - STRIDES (list): List of stride values corresponding to feature map scales.
    - ANCHORS (list): Anchor box dimensions for each scale.
    - i (int, optional): The index of the current feature map scale (default: 0).
    - XYSCALE (list, optional): Scaling factors for bounding box center adjustments (default: [1, 1, 1]).

    Returns:
    - pred_xywh (tf.Tensor): Bounding box coordinates with shape (batch_size, num_boxes, 4),
      where each box is represented as [x_center, y_center, width, height].
    - pred_prob (tf.Tensor): Class probabilities with shape (batch_size, num_boxes, NUM_CLASS).
    """
    batch_size = tf.shape(conv_output)[0]
    conv_output = tf.reshape(
        conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS)
    )

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(
        conv_output, (2, 2, 1, NUM_CLASS), axis=-1
    )

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])

    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (
        tf.reshape(tf.sigmoid(conv_raw_dxdy), (-1, 2)) * XYSCALE[i]
        - 0.5 * (XYSCALE[i] - 1)
        + tf.reshape(xy_grid, (-1, 2))
    ) * STRIDES[i]
    pred_xy = tf.reshape(pred_xy, (batch_size, output_size, output_size, 3, 2))
    pred_wh = tf.exp(conv_raw_dwdh) * ANCHORS[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    pred_prob = pred_conf * pred_prob

    pred_prob = tf.reshape(pred_prob, (batch_size, -1, NUM_CLASS))
    pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))
    return pred_xywh, pred_prob


def filter_boxes(
    box_xywh, scores, score_threshold=0.4, input_shape=tf.constant([416, 416])
):
    """
    Filters bounding boxes and confidence scores based on a threshold and normalizes coordinates.

    Parameters:
    - box_xywh (tf.Tensor): Bounding boxes with shape (batch_size, num_boxes, 4),
      where each box is represented as [x_center, y_center, width, height].
    - scores (tf.Tensor): Confidence scores for each bounding box, with shape
      (batch_size, num_boxes, num_classes).
    - score_threshold (float, optional): Threshold for filtering low-confidence boxes
      (default: 0.4).
    - input_shape (tf.Tensor, optional): Shape of the input image as [height, width]
      (default: [416, 416]).

    Returns:
    - boxes (tf.Tensor): Normalized bounding boxes with shape
      (batch_size, num_filtered_boxes, 4), where each box is represented as
      [y_min, x_min, y_max, x_max].
    - pred_conf (tf.Tensor): Filtered confidence scores with shape
      (batch_size, num_filtered_boxes, num_classes).
    """
    scores_max = tf.math.reduce_max(scores, axis=-1)

    mask = scores_max >= score_threshold
    class_boxes = tf.boolean_mask(box_xywh, mask)
    pred_conf = tf.boolean_mask(scores, mask)
    class_boxes = tf.reshape(
        class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]]
    )
    pred_conf = tf.reshape(
        pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]]
    )

    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

    input_shape = tf.cast(input_shape, dtype=tf.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.0)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.0)) / input_shape
    boxes = tf.concat(
        [
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2],  # x_max
        ],
        axis=-1,
    )
    return (boxes, pred_conf)


def compute_loss(pred, conv, label, bboxes, STRIDES, NUM_CLASS, IOU_LOSS_THRESH, i=0):
    """
    Computes the total loss for YOLO training, including GIoU loss, confidence loss,
    and classification probability loss.

    Parameters:
    - pred (tf.Tensor): Decoded bounding box predictions with shape
      (batch_size, grid_size, grid_size, 3, 5 + NUM_CLASS), containing:
        - [x_center, y_center, width, height, object_confidence, class_probabilities].
    - conv (tf.Tensor): Raw output from the convolutional layer before decoding.
    - label (tf.Tensor): Ground truth labels with shape
      (batch_size, grid_size, grid_size, 3, 5 + NUM_CLASS), containing:
        - [x_center, y_center, width, height, object_presence, one_hot_class_probabilities].
    - bboxes (tf.Tensor): Ground truth bounding boxes with shape
      (batch_size, num_boxes, 4), formatted as [x_min, y_min, x_max, y_max].
    - STRIDES (list): List of stride values corresponding to feature map scales.
    - NUM_CLASS (int): Number of object classes in the dataset.
    - IOU_LOSS_THRESH (float): IoU threshold for determining background versus foreground boxes.
    - i (int, optional): Index of the current feature map scale (default: 0).

    Returns:
    - giou_loss (tf.Tensor): Generalized Intersection over Union (GIoU) loss.
    - conf_loss (tf.Tensor): Confidence loss for object presence detection.
    - prob_loss (tf.Tensor): Classification loss for predicting the correct object class.
    """
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]

    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    giou = tf.expand_dims(utils.bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[
        :, :, :, :, 3:4
    ] / (input_size**2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    iou = utils.bbox_iou(
        pred_xywh[:, :, :, :, np.newaxis, :],
        bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :],
    )
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < IOU_LOSS_THRESH, tf.float32)

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
        respond_bbox
        * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=respond_bbox, logits=conv_raw_conf
        )
        + respond_bgd
        * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=respond_bbox, logits=conv_raw_conf
        )
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(
        labels=label_prob, logits=conv_raw_prob
    )

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return giou_loss, conf_loss, prob_loss
