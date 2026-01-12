import numpy as np

from mlops.shapes.typing.bboxes import *
from mlops.shapes.typing.polys import *


__all__ = [
    "polyLabelme_to_bboxLabelme",
    "polyCoco_to_bboxCoco",
    "polyYolo_to_bboxYolo"
]


def polyLabelme_to_bboxLabelme(
    poly: PolyLabelmeType
) -> BBoxLabelmeType:
    poly = np.asarray(poly)
    x1 = np.minimum(poly[:, 0])
    y1 = np.minimum(poly[:, 1])
    x2 = np.maximum(poly[:, 0])
    y2 = np.maximum(poly[:, 1])
    bbox = [x1, y1, x2, y2]
    return bbox

def polyCoco_to_bboxCoco(
    poly: PolyCocoType
) -> BBoxCocoType:
    poly = np.asarray(poly).reshape(-1, 2)
    x1 = np.minimum(poly[:, 0])
    y1 = np.minimum(poly[:, 1])
    x2 = np.maximum(poly[:, 0])
    y2 = np.maximum(poly[:, 1])
    w = x2 - x1
    h = y2 - y1
    bbox = [x1, y1, w, h]
    return bbox

def polyYolo_to_bboxYolo(
    poly: PolyYoloType
) -> BBoxYoloType:
    poly = np.asarray(poly).reshape(-1, 2)
    x1_norm = np.minimum(poly[:, 0])
    y1_norm = np.minimum(poly[:, 1])
    x2_norm = np.maximum(poly[:, 0])
    y2_norm = np.maximum(poly[:, 1])
    w_norm = x2_norm - x1_norm
    h_norm = y2_norm - y1_norm
    x_ctr_norm = (x1_norm + x2_norm) / 2
    y_ctr_norm = (y1_norm + y2_norm) / 2
    bbox = [x_ctr_norm, y_ctr_norm, w_norm, h_norm]
    return bbox

