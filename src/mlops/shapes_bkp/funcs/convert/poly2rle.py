from typing import Tuple, List

import numpy as np
import pycocotools.mask as pycocomask

from mlops.shapes.typing.polys import *
from mlops.shapes.typing.rles import RleType


__all__ = [
    "polyArrs_to_rles",
    "polyLabelme_to_rle",
    "polysLabelme_to_rles",
    "polysCoco_to_rles",
    "polysYolo_to_rles",
]


def polyArrs_to_rles(
    polys: List[PolyArrType],
    img_hw: Tuple[int, int],
    merge_flag: bool
) -> List[RleType]:
    polys = [p.flatten().astype(np.int32) for p in polys]
    rles = pycocomask.frPyObjects(polys, img_hw[0], img_hw[1])

    if merge_flag:
        rle_merge = pycocomask.merge(rles)
        rles = [rle_merge]
    
    return rles

def polysLabelme_to_rles(
    polys: List[PolyLabelmeType],
    img_hw: Tuple[int, int],
    merge_flag: bool
) -> List[RleType]:
    polys = [np.asarray(p).astype(np.int32).flatten().tolist() for p in polys]
    rles = pycocomask.frPyObjects(polys, img_hw[0], img_hw[1])

    if merge_flag:
        rle_merge = pycocomask.merge(rles)
        rles = [rle_merge]
    
    return rles

def polysCoco_to_rles(
    polys: List[PolyCocoType],
    img_hw: Tuple[int, int],
    merge_flag: bool
) -> List[RleType]:
    polys = [np.asarray(p).astype(np.int32).tolist() for p in polys]
    rles = pycocomask.frPyObjects(polys, img_hw[0], img_hw[1])

    if merge_flag:
        rle_merge = pycocomask.merge(rles)
        rles = [rle_merge]
    
    return rles

def polysYolo_to_rles(
    polys: List[PolyYoloType],
    img_hw: Tuple[int, int],
    merge_flag: bool
) -> List[RleType]:
    img_h, img_w = img_hw
    polys_ = []
    for poly in polys:
        poly = np.asarray(poly).reshape(-1, 2)
        poly[:, 0] = poly[:, 0] * img_w
        poly[:, 1] = poly[:, 1] * img_h
        poly = poly.astype(np.int32).flatten().tolist()
        polys_.append(poly)
    polys = polys_

    rles = pycocomask.frPyObjects(polys, img_hw[0], img_hw[1])

    if merge_flag:
        rle_merge = pycocomask.merge(rles)
        rles = [rle_merge]
    
    return rles

def polyLabelme_to_rle(
    poly: PolyLabelmeType,
    img_hw: Tuple[int, int],
) -> RleType:
    poly = [np.asarray(poly).astype(np.int32).tolist()]
    rle = pycocomask.frPyObjects(poly, img_hw[0], img_hw[1])[0]
    return rle
