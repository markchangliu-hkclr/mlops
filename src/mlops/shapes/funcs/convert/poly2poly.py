from typing import Tuple

import numpy as np

from mlops.shapes.typing.polys import *


__all__ = [
    "polyArr_to_polyLabelme",
    "polyArr_to_polyCoco",
    "polyArr_to_polyYolo",
    "polyLabelme_to_polyArr",
    "polyLabelme_to_polyCoco",
    "polyLabelme_to_polyYolo",
]


def polyArr_to_polyLabelme(
    poly: PolyArrType
) -> PolyLabelmeType:
    poly = poly.tolist()
    return poly

def polyArr_to_polyCoco(
    poly: PolyArrType
) -> PolyCocoType:
    poly = polyArr_to_polyLabelme(poly)
    poly = polyLabelme_to_polyCoco(poly)
    return poly

def polyArr_to_polyYolo(
    poly: PolyArrType,
    img_hw: Tuple[int, int]
) -> PolyYoloType:
    poly = polyArr_to_polyLabelme(poly)
    poly = polyLabelme_to_polyYolo(poly, img_hw)
    return poly

def polyLabelme_to_polyCoco(
    poly: PolyLabelmeType
) -> PolyCocoType:
    poly = np.asarray(poly).flatten().tolist()
    return poly

def polyLabelme_to_polyYolo(
    poly: PolyLabelmeType,
    img_hw: Tuple[int, int]
) -> PolyYoloType:
    img_h, img_w = img_hw
    poly = np.asarray(poly)
    poly[:, 0] = poly[:, 0] / img_w
    poly[:, 1] = poly[:, 1] / img_h
    poly = poly.flatten()
    return poly

def polyLabelme_to_polyArr(
    poly: PolyLabelmeType
) -> PolyArrType:
    poly = np.asarray(poly)
    return poly
