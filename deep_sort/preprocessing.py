# vim: expandtab:ts=4:sw=4
import numpy as np


def non_max_suppression(boxes, classes, max_bbox_overlap, scores=None):
    """Suppress overlapping detections.

    Original code from [1]_ has been adapted to include confidence score.

    .. [1] http://www.pyimagesearch.com/2015/02/16/
           faster-non-maximum-suppression-python/

    Examples
    --------

        >>> boxes = [d.roi for d in detections]
        >>> classes = [d.classes for d in detections]
        >>> scores = [d.confidence for d in detections]
        >>> indices = non_max_suppression(boxes, max_bbox_overlap, scores)
        >>> detections = [detections[i] for i in indices]

    Parameters
    ----------
    boxes : ndarray
        Array of ROIs (x, y, width, height).
    classes : Optional[array_like]
        Class labels; overlapping boxes of different classes are never
        suppressed against each other.
    max_bbox_overlap : float
        ROIs that overlap more than this values are suppressed.
    scores : Optional[array_like]
        Detector confidence score.

    Returns
    -------
    List[int]
        Returns indices of detections that have survived non-maxima suppression.

    """
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float64)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]
    y2 = boxes[:, 3] + boxes[:, 1]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if scores is not None:
        scores = np.asarray(scores)
        if len(scores) != len(boxes):
            raise ValueError("scores and boxes must have the same length")
    else:
        scores = y2

    if classes is None:
        class_groups = [np.arange(len(boxes))]
    else:
        class_values = np.asarray(classes)
        if len(class_values) != len(boxes):
            raise ValueError("classes and boxes must have the same length")
        class_groups = [
            np.flatnonzero(class_values == class_value)
            for class_value in np.unique(class_values)
        ]

    pick = []
    for class_indices in class_groups:
        idxs = class_indices[np.argsort(scores[class_indices])]

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(
                idxs, np.concatenate(([last], np.where(overlap > max_bbox_overlap)[0]))
            )

    return [
        int(index)
        for index in sorted(
            pick, key=lambda index: scores[index], reverse=True
        )
    ]
