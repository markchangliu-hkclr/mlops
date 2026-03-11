from typing import Tuple, List

import numpy as np
import pycocotools.mask as pycocomask

from mlops.datasets.types import PolyArrType, PolyCocoType, PolyLabelmeType, PolyYoloType
from mlops.datasets.types import RleType
from mlops.datasets.types import MaskArrType


def convert_polyArr2polyLabelme(
    poly: PolyArrType
) -> PolyLabelmeType:
    poly = poly.tolist()
    return poly

def convert_polyArr2polyCoco(
    poly: PolyArrType
) -> PolyCocoType:
    poly = poly.flatten().tolist()
    return poly

def convert_polyArr2polyYolo(
    poly: PolyArrType,
    img_hw: Tuple[int, int]
) -> PolyYoloType:
    poly = poly.copy()
    poly[:, 0] = poly[:, 0] / img_hw[0]
    poly[:, 1] = poly[:, 1] / img_hw[1]
    poly = poly.flatten().tolist()
    return poly

def convert_polyLabelme2polyArr(
    poly: PolyLabelmeType
) -> PolyArrType:
    poly = np.asarray(poly)
    return poly

def convert_polyLabelme2polyCoco(
    poly: PolyLabelmeType
) -> PolyCocoType:
    poly = convert_polyLabelme2polyArr(poly)
    poly = convert_polyArr2polyCoco(poly)
    return poly

def convert_polyLabelme2polyYolo(
    poly: PolyLabelmeType,
    img_hw: Tuple[int, int]
) -> PolyYoloType:
    img_h, img_w = img_hw
    poly = np.asarray(poly)
    poly[:, 0] = poly[:, 0] / img_w
    poly[:, 1] = poly[:, 1] / img_h
    poly = poly.flatten()
    return poly

def convert_polyArr2rle(
    polys: List[PolyArrType],
    img_hw: Tuple[int, int],
    merge_flag: bool
) -> List[RleType]:
    polys = [np.astype(p, np.double) for p in polys]
    rles = pycocomask.frPyObjects(polys, img_hw[0], img_hw[1])

    if merge_flag:
        rles = [pycocomask.merge(rles)]

    return rles

def convert_polyArr2maskArr(
    polys: List[PolyArrType],
    img_hw: Tuple[int, int],
    merge_flag: bool
) -> List[MaskArrType]:
    rles = convert_polyArr2rle(polys, img_hw, merge_flag)
    masks = []

    for rle in rles:
        mask = pycocomask.decode(rle)
        masks.append(mask)
    
    return masks