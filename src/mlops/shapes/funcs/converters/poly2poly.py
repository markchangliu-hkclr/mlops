from typing import Tuple

import numpy as np

import mlops.shapes.typing.poly as poly_type


__all__ = [
    "poly2poly_labelme2coco",
    "poly2poly_coco2yolo",
    "poly2poly_labelme2yolo"
]


def poly2poly_labelme2coco(
    poly: poly_type.PolyLabelmeType
) -> poly_type.PolyCocoType:
    poly = np.asarray(poly).flatten().tolist()
    return poly

def poly2poly_coco2yolo(
    poly: poly_type.PolyCocoType,
    img_hw: Tuple[int, int]
) -> poly_type.PolyYoloType:
    poly = np.asarray(poly).reshape(-1, 2)
    img_wh = np.asarray((img_hw[1], img_hw[0])).reshape(-1, 2)
    poly = (poly / img_wh).flatten().tolist()
    return poly

def poly2poly_labelme2yolo(
    poly: poly_type.PolyLabelmeType,
    img_hw: Tuple[int, int]
) -> poly_type.PolyYoloType:
    poly = np.asarray(poly)
    img_wh = np.asarray((img_hw[1], img_hw[0])).reshape(-1, 2)
    poly = (poly / img_wh).flatten().tolist()
    return poly