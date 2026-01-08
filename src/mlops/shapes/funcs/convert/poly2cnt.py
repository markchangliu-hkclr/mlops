from typing import Tuple

import numpy as np

import mlops.shapes.typing.cnt as cnt_type
import mlops.shapes.typing.poly as poly_type


__all__ = [
    "poly2cnt_labelme",
    "poly2cnt_coco",
    "poly2cnt_yolo"
]


def poly2cnt_labelme(
    poly: poly_type.PolyLabelmeType
) -> cnt_type.ContourType:
    cnt = np.asarray(poly, dtype = np.int32)
    cnt = np.expand_dims(cnt, 1)
    return cnt

def poly2cnt_coco(
    poly: poly_type.PolyCocoType
) -> cnt_type.ContourType:
    cnt = np.asarray(poly, dtype = np.int32).reshape(-1, 1, 2)
    return cnt

def poly2cnt_yolo(
    poly: poly_type.PolyYoloType,
    img_hw: Tuple[int, int]
) -> cnt_type.ContourType:
    cnt = np.asarray(poly, dtype = np.int32).reshape(-1, 1, 2)
    img_wh = np.asarray((img_hw[1], img_hw[0])).reshape(1, 1, 2)
    cnt = cnt * img_wh
    return cnt