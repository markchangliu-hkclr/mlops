from typing import Tuple

import numpy as np

# import mlops.shapes.typing.cnt as cnt_type
# import mlops.shapes.typing.poly as poly_type

from mlops.shapes.typing.polys import *
from mlops.shapes.typing.contours import *


__all__ = [
    "polyArr_to_contour",
    "polyLabelme_to_contour",
    "polyCoco_to_contour",
    "polyYolo_to_contour"
]


def polyArr_to_contour(
    poly: PolyArrType
) -> ContourType:
    cnt = poly[:, None, :].astype(np.int32)
    return cnt

def polyLabelme_to_contour(
    poly: PolyLabelmeType
) -> ContourType:
    cnt = np.asarray(poly)[:, None, :].astype(np.int32)
    return cnt

def polyCoco_to_contour(
    poly: PolyCocoType
) -> ContourType:
    cnt = np.asarray(poly).reshape((-1, 1, 2)).astype(np.int32)
    return cnt

def polyYolo_to_contour(
    poly: PolyYoloType,
    img_hw: Tuple[int, int]
) -> ContourType:
    img_h, img_w = img_hw
    poly = np.asarray(poly).reshape((-1, 1, 2))
    poly[:, :, 0] = poly[:, :, 0] * img_w
    poly[:, :, 1] = poly[:, :, 1] * img_h
    poly = poly.astype(np.int32)
    return poly

