from typing import Tuple

import numpy as np
import pycocotools.mask as pycocomask

import mlops.shapes.typing.poly as poly_type
import mlops.shapes.typing.rle as rle_type


__all__ = [
    "poly2rle_labelme",
    "poly2rle_coco",
    "poly2rle_yolo",
    "polys2rle_merge_coco"
]


def poly2rle_labelme(
    poly: poly_type.PolyLabelmeType,
    img_hw: Tuple[int, int]
) -> rle_type.RLEType:
    poly = np.asarray(poly, dtype = np.int32).flatten().tolist()
    rle = pycocomask.frPyObjects([poly], img_hw[0], img_hw[1])[0]
    return rle

def poly2rle_coco(
    poly: poly_type.PolyCocoType,
    img_hw: Tuple[int, int]
) -> rle_type.RLEType:
    rle = pycocomask.frPyObjects([poly], img_hw[0], img_hw[1])[0]
    return rle

def poly2rle_yolo(
    poly: poly_type.PolyYoloType,
    img_hw: Tuple[int, int]
) -> rle_type.RLEType:
    poly = np.asarray(poly).reshape(-1, 2)
    img_wh = np.asarray((img_hw[1], img_hw[0])).reshape(-1, 2)
    poly = (poly * img_wh).astype(np.int32).flatten().tolist()
    rle = pycocomask.frPyObjects([poly], img_hw[0], img_hw[1])[0]
    return rle

def polys2rle_merge_coco(
    polys: poly_type.PolysCocoType,
    img_hw: Tuple[int, int]
) -> rle_type.RLEType:
    rles = pycocomask.frPyObjects(polys, img_hw[0], img_hw[1])
    rle = pycocomask.merge(rles)
    return rle