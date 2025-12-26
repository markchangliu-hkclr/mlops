from typing import Tuple

import numpy as np

from mlops.shapes.typing import ContourType, PolyLabelmeType, \
    PolyCocoType, PolyYoloType, ContoursType, PolysLabelmeType, \
    PolysCocoType, PolysYoloType


__all__ = [
    "cnt2poly_labelme",
    "cnt2poly_coco",
    "cnt2poly_yolo",
    "cnts2polys_labelme",
    "cnts2polys_coco",
    "cnts2polys_yolo"
]


def cnt2poly_labelme(
    cnt: ContourType
) -> PolyLabelmeType:
    poly_labelme = np.squeeze(cnt).tolist()
    return poly_labelme

def cnt2poly_coco(
    cnt: ContourType
) -> PolyCocoType:
    poly_coco = cnt.flatten().tolist()
    return poly_coco

def cnt2poly_yolo(
    cnt: ContourType,
    img_hw: Tuple[int, int]
) -> PolyYoloType:
    poly_yolo = np.squeeze(cnt)
    poly_yolo = poly_yolo / np.asarray((img_hw[1], img_hw[0])).reshape(-1, 2)
    poly_yolo = poly_yolo.flatten().tolist()
    return poly_yolo

def cnts2polys_labelme(
    cnts: ContoursType
) -> PolysLabelmeType:
    polys_labelme = []

    for cnt in cnts:
        polys_labelme = cnt2poly_labelme(cnt)
        polys_labelme.append(polys_labelme)

    return polys_labelme

def cnts2polys_coco(
    cnts: ContoursType
) -> PolysCocoType:
    polys_coco = []

    for cnt in cnts:
        poly_coco = cnt2poly_coco(cnt)
        polys_coco.append(poly_coco)
    
    return polys_coco

def cnts2polys_yolo(
    cnts: ContoursType,
    img_hw: Tuple[int, int]
) -> PolysYoloType:
    polys_yolo = []

    for cnt in cnts:
        poly_yolo = cnt2poly_yolo(cnt, img_hw)
        polys_yolo.append(poly_yolo)
    
    return polys_yolo
    