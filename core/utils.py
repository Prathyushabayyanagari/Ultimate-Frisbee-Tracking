import colorsys
import random

import cv2
import numpy as np
import tensorflow as tf

from core.config import cfg


def load_freeze_layer(model="yolov4", tiny=False):
    """
    Identifies the layers to freeze during training based on the model type.

    Parameters:
    - model (str): The model name, either 'yolov3' or 'yolov4' (default: 'yolov4').
    - tiny (bool): Indicates whether to use the Tiny version of the model (default: False).

    Returns:
    - freeze_layouts (list): A list of layer names to be frozen during training.
    """
    if tiny:
        if model == "yolov3":
            freeze_layouts = ["conv2d_9", "conv2d_12"]
        else:
            freeze_layouts = ["conv2d_17", "conv2d_20"]
    else:
        if model == "yolov3":
            freeze_layouts = ["conv2d_58", "conv2d_66", "conv2d_74"]
        else:
            freeze_layouts = ["conv2d_93", "conv2d_101", "conv2d_109"]
    return freeze_layouts


def load_weights(model, weights_file, model_name="yolov4", tiny=False):
    """
    Loads pre-trained weights into the YOLO model from a Darknet weights file.

    Parameters:
    - model (tf.keras.Model): The YOLO model to load weights into.
    - weights_file (str): Path to the Darknet weights file.
    - model_name (str): The model name, either 'yolov3' or 'yolov4' (default: 'yolov4').
    - tiny (bool): Indicates whether the model is a Tiny version (default: False).

    Notes:
    - Adjusts weights to match TensorFlow format (e.g., shape and layer names).
    - Assigns weights to both convolutional layers and batch normalization layers, depending
      on the layer type.
    - Handles special cases for the output layers, which include biases in the weights.

    """
    if tiny:
        if model_name == "yolov3":
            layer_size = 13
            output_pos = [9, 12]
        else:
            layer_size = 21
            output_pos = [17, 20]
    else:
        if model_name == "yolov3":
            layer_size = 75
            output_pos = [58, 66, 74]
        else:
            layer_size = 110
            output_pos = [93, 101, 109]
    wf = open(weights_file, "rb")
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(layer_size):
        conv_layer_name = f"conv2d_{i}" if i > 0 else "conv2d"
        bn_layer_name = f"batch_normalization_{j}" if j > 0 else "batch_normalization"

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in output_pos:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in output_pos:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    # assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def read_class_names(class_file_name):
    """
    Reads class names from a file and maps them to numerical IDs.

    Parameters:
    - class_file_name (str): The path to the file containing class names. Each line in
      the file represents a class name.

    Returns:
    - names (dict): A dictionary where keys are numerical IDs (starting from 0) and
      values are the corresponding class names.
    """
    names = {}
    with open(class_file_name) as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip("\n")
    return names


def load_config(FLAGS):
    """
    Loads YOLO configuration parameters based on the provided FLAGS.

    Parameters:
    - FLAGS (Namespace): A collection of flags specifying model settings. Relevant attributes:
        - FLAGS.tiny (bool): Indicates whether the Tiny YOLO version is used.
        - FLAGS.model (str): Specifies the YOLO model type ('yolov3' or 'yolov4').

    Returns:
    - STRIDES (np.ndarray): An array of strides used in the YOLO model for feature map scaling.
    - ANCHORS (list): A list of anchor box dimensions, determined based on model type and size.
    - NUM_CLASS (int): The number of classes in the dataset, derived from the class names file.
    - XYSCALE (list): A list of scaling factors for adjusting bounding box predictions, depending
      on the model type and size.
    """
    if FLAGS.tiny:
        STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        ANCHORS = get_anchors(cfg.YOLO.ANCHORS_TINY, FLAGS.tiny)
        XYSCALE = cfg.YOLO.XYSCALE_TINY if FLAGS.model == "yolov4" else [1, 1]
    else:
        STRIDES = np.array(cfg.YOLO.STRIDES)
        if FLAGS.model == "yolov4":
            ANCHORS = get_anchors(cfg.YOLO.ANCHORS, FLAGS.tiny)
        elif FLAGS.model == "yolov3":
            ANCHORS = get_anchors(cfg.YOLO.ANCHORS_V3, FLAGS.tiny)
        XYSCALE = cfg.YOLO.XYSCALE if FLAGS.model == "yolov4" else [1, 1, 1]
    NUM_CLASS = len(read_class_names(cfg.YOLO.CLASSES))

    return STRIDES, ANCHORS, NUM_CLASS, XYSCALE


def get_anchors(anchors_path, tiny=False):
    """
    Reshapes anchor box dimensions for YOLO models based on the model size.

    Parameters:
    - anchors_path (list or np.ndarray): A list or array containing anchor box dimensions.
      Each anchor is typically represented as [width, height].
    - tiny (bool): Indicates whether to format the anchors for the Tiny YOLO version (default: False).

    Returns:
    - anchors (np.ndarray): A NumPy array of reshaped anchor box dimensions with:
        - Shape (2, 3, 2) for Tiny YOLO models (2 feature map scales, 3 anchors per scale).
        - Shape (3, 3, 2) for full-sized YOLO models (3 feature map scales, 3 anchors per scale).
    """
    anchors = np.array(anchors_path)
    if tiny:
        return anchors.reshape(2, 3, 2)
    else:
        return anchors.reshape(3, 3, 2)


def image_preprocess(image, target_size, gt_boxes=None):
    """
    Resizes an image to the target size while maintaining aspect ratio, and applies padding.

    Parameters:
    - image (np.ndarray): The input image as a NumPy array (shape: [height, width, 3]).
    - target_size (tuple): The desired output size as (height, width).
    - gt_boxes (np.ndarray, optional): Ground-truth bounding boxes to be adjusted,
      formatted as [x_min, y_min, x_max, y_max].

    Returns:
    - image_paded (np.ndarray): The preprocessed image scaled to [target_size[0], target_size[1], 3].
    - gt_boxes (np.ndarray, optional): The adjusted ground-truth bounding boxes (if provided),
      resized and padded to match the processed image.
    """
    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh : nh + dh, dw : nw + dw, :] = image_resized
    image_paded = image_paded / 255.0

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def format_boxes(bboxes, image_height, image_width):
    """
    Converts bounding boxes from normalized coordinates to pixel-based format.

    Parameters:
    - bboxes (np.ndarray): Normalized bounding boxes with values between 0 and 1,
      formatted as [ymin, xmin, ymax, xmax].
    - image_height (int): The height of the original image in pixels.
    - image_width (int): The width of the original image in pixels.

    Returns:
    - bboxes (np.ndarray): Bounding boxes converted to pixel format, formatted as
      [xmin, ymin, width, height].
    """
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        width = xmax - xmin
        height = ymax - ymin
        box[0], box[1], box[2], box[3] = xmin, ymin, width, height
    return bboxes


def draw_bbox(
    image,
    bboxes,
    info=False,
    show_label=True,
    classes=read_class_names(cfg.YOLO.CLASSES),
):
    """
    Draws bounding boxes on an image and optionally includes class labels and confidence scores.

    Parameters:
    - image (np.ndarray): The input image on which bounding boxes will be drawn.
    - bboxes (tuple): Contains bounding box information as (out_boxes, out_scores, out_classes, num_boxes):
        - out_boxes (list): List of bounding boxes, formatted as [x, y, width, height].
        - out_scores (list): Confidence scores for each bounding box.
        - out_classes (list): Class indices for each bounding box.
        - num_boxes (int): Total number of bounding boxes.
    - info (bool, optional): If True, prints object details (class, confidence, and coordinates) to the console.
    - show_label (bool, optional): If True, includes class labels and confidence scores in the annotations.
    - classes (dict, optional): Dictionary mapping class indices to class names. Defaults to `read_class_names()`.

    Returns:
    - image (np.ndarray): The annotated image with bounding boxes, labels, and scores (if applicable).
    """
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1.0, 1.0) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors)
    )

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_scores, out_classes, num_boxes = bboxes
    for i in range(num_boxes):
        if int(out_classes[i]) < 0 or int(out_classes[i]) > num_classes:
            continue
        x, y, w, h = out_boxes[i]
        fontScale = 0.5
        score = out_scores[i]
        class_ind = int(out_classes[i])
        class_name = classes[class_ind]
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (x, y), (x + w, y + h)
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if info:
            print(
                f"Object found: {class_name}, Confidence: {score:.2f}, BBox Coords (xmin, ymin, width, height): {x}, {y}, {w}, {h} "
            )

        if show_label:
            bbox_mess = f"{class_name}: {score:.2f}"
            t_size = cv2.getTextSize(
                bbox_mess, 0, fontScale, thickness=bbox_thick // 2
            )[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(
                image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1
            )  # filled

            cv2.putText(
                image,
                bbox_mess,
                (c1[0], np.float32(c1[1] - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale,
                (0, 0, 0),
                bbox_thick // 2,
                lineType=cv2.LINE_AA,
            )
    return image


def bbox_iou(bboxes1, bboxes2):
    """
    Calculates the Intersection over Union (IoU) between two sets of bounding boxes.

    Parameters:
    - bboxes1 (tf.Tensor): Tensor representing the first set of bounding boxes with shape (..., 4),
      where the last dimension is [x_center, y_center, width, height].
    - bboxes2 (tf.Tensor): Tensor representing the second set of bounding boxes with shape (..., 4).

    Returns:
    - iou (tf.Tensor): Tensor of IoU values, with shape based on the input tensors.
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    return iou


def bbox_giou(bboxes1, bboxes2):
    """
    Calculates the Generalized Intersection over Union (GIoU) between two sets of bounding boxes.

    Parameters:
    - bboxes1 (tf.Tensor): Tensor of bounding boxes with shape (..., 4), where the last dimension
      is [x_center, y_center, width, height].
    - bboxes2 (tf.Tensor): Tensor of bounding boxes with shape (..., 4).

    Returns:
    - giou (tf.Tensor): Tensor containing the GIoU values, with a shape based on the input tensors.
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tf.maximum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    enclose_section = enclose_right_down - enclose_left_up
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1]

    giou = iou - tf.math.divide_no_nan(enclose_area - union_area, enclose_area)

    return giou


def bbox_ciou(bboxes1, bboxes2):
    """
    Calculates the Complete Intersection over Union (CIoU) between two sets of bounding boxes.

    Parameters:
    - bboxes1 (tf.Tensor): Tensor of bounding boxes with shape (..., 4), where the last dimension
      is [x_center, y_center, width, height].
    - bboxes2 (tf.Tensor): Tensor of bounding boxes with shape (..., 4).

    Returns:
    - ciou (tf.Tensor): Tensor containing the CIoU values, with a shape based on the input tensors.
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tf.maximum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    enclose_section = enclose_right_down - enclose_left_up

    c_2 = enclose_section[..., 0] ** 2 + enclose_section[..., 1] ** 2

    center_diagonal = bboxes2[..., :2] - bboxes1[..., :2]

    rho_2 = center_diagonal[..., 0] ** 2 + center_diagonal[..., 1] ** 2

    diou = iou - tf.math.divide_no_nan(rho_2, c_2)

    v = (
        (
            tf.math.atan(tf.math.divide_no_nan(bboxes1[..., 2], bboxes1[..., 3]))
            - tf.math.atan(tf.math.divide_no_nan(bboxes2[..., 2], bboxes2[..., 3]))
        )
        * 2
        / np.pi
    ) ** 2

    alpha = tf.math.divide_no_nan(v, 1 - iou + v)

    ciou = diou - alpha * v

    return ciou


def nms(bboxes, iou_threshold, sigma=0.3, method="nms"):
    """
    Applies Non-Maximum Suppression (NMS) or Soft-NMS to filter bounding boxes.

    Parameters:
    - bboxes (np.ndarray): Bounding boxes, where each box is represented as
      [xmin, ymin, xmax, ymax, score, class].
    - iou_threshold (float): The IoU threshold for suppressing overlapping boxes in NMS.
    - sigma (float, optional): The sigma value for Soft-NMS (default: 0.3).
    - method (str, optional): The suppression method, either "nms" or "soft-nms" (default: "nms").

    Returns:
    - best_bboxes (list): A list of filtered bounding boxes after applying NMS or Soft-NMS.
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = bboxes[:, 5] == cls
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate(
                [cls_bboxes[:max_ind], cls_bboxes[max_ind + 1 :]]
            )
            iou = bbox_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ["nms", "soft-nms"]

            if method == "nms":
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == "soft-nms":
                weight = np.exp(-(1.0 * iou**2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.0
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)


def unfreeze_all(model, frozen=False):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            unfreeze_all(l, frozen)
