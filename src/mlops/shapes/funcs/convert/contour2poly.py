from typing import Tuple, List

import numpy as np

from mlops.shapes.typing import (
    PolyArrType,
    PolyLabelmeType,
    PolyCocoType,
    PolyYoloType,
    ContourType,
)


__all__ = [
    "contour_to_polyArr",
    "contour_to_polyLabelme",
    "contour_to_polyCoco",
    "contour_to_polyYolo",

    "contours_to_polyArrs",
    "contours_to_polyLabelmes",
    "contours_to_polyCocos",
    "contours_to_polyYolos"
]


def contour_to_polyArr(
    contour: ContourType
) -> PolyArrType:
    poly = np.squeeze(contour, axis = 1)
    return poly

def contour_to_polyLabelme(
    contour: ContourType
) -> PolyLabelmeType:
    poly = np.squeeze(contour, axis = 1).tolist()
    return poly

def contour_to_polyCoco(
    contour: ContourType,
) -> PolyCocoType:
    poly = contour.flatten().tolist()
    return poly

def contour_to_polyYolo(
    contour: ContourType,
    img_hw: Tuple[int, int]
) -> PolyYoloType:
    img_h, img_w = img_hw
    poly = contour.copy()
    poly[:, :, 0] = poly[:, :, 0] / img_w
    poly[:, :, 1] = poly[:, :, 1] / img_h
    poly = poly.flatten().tolist()
    return poly

def contours_to_polyArrs(
    contours: List[ContourType]
) -> List[PolyArrType]:
    polys = []
    for cnt in contours:
        poly = contour_to_polyArr(cnt)
        polys.append(poly)
    return polys

def contours_to_polyLabelmes(
    contours: List[ContourType]
) -> List[PolyLabelmeType]:
    polys = []
    for cnt in contours:
        poly = contour_to_polyLabelme(cnt)
        polys.append(poly)
    return polys 

def contours_to_polyCocos(
    contours: List[ContourType]
) -> List[PolyCocoType]:
    polys = []
    for cnt in contours:
        poly = contour_to_polyCoco(cnt)
        polys.append(poly)
    return polys 

def contours_to_polyYolos(
    contours: List[ContourType],
    img_hw: Tuple[int, int]
) -> List[PolyYoloType]:
    polys = []
    for cnt in contours:
        poly = contour_to_polyYolo(cnt, img_hw)
        polys.append(poly)
    return polys

