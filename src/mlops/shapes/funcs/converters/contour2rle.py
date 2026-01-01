import os
from typing import Union, Tuple

import pycocotools.mask as pycocomask

import mlops.shapes.funcs.cnt2poly as cnt2poly
import mlops.shapes.typing.cnt as cnt_type
import mlops.shapes.typing.rle as rle_type


__all__ = [
    "cnt2rle",
    "cnts2rles",
    "cnts2rle_merge"
]


def cnt2rle(
    cnt: cnt_type.ContourType,
    img_hw: Tuple[int, int]
) -> rle_type.RLEType:
    poly_coco = cnt2poly.cnt2poly_coco(cnt)

    if len(poly_coco) == 4:
        poly_coco += poly_coco[-2:]

    rle = pycocomask.frPyObjects([poly_coco], img_hw[0], img_hw[1])[0]
    return rle

def cnts2rles(
    cnts: cnt_type.ContoursType,
    img_hw: Tuple[int, int]
) -> rle_type.RLEsType:
    polys_coco = cnt2poly.cnts2polys_coco(cnts)
    rles = pycocomask.frPyObjects(polys_coco, img_hw[0], img_hw[1])
    return rles

def cnts2rle_merge(
    cnts: cnt_type.ContoursType,
    img_hw: Tuple[int, int]
) -> rle_type.RLEType:
    rles = cnts2rles(cnts, img_hw)
    rle_merge = pycocomask.merge(rles, False)
    return rle_merge