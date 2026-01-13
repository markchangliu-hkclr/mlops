from typing import Tuple

import numpy as np

from mlops.shapes.typing.bboxes import *


__all__ = [
    "bboxesArrXYXY_to_bboxesArrXYWH",
    "bboxesArrXYWH_to_bboxesArrXYXY",
    "bboxesArrXYXY_to_bboxesLabelme",
    "bboxesArrXYXY_to_bboxesCoco",
    "bboxesArrXYXY_to_bboxesYolo",
    "bboxesLabelme_to_bboxesArrXYXY",
    "bboxesCoco_to_bboxesArrXYXY",
    "bboxesYolo_to_bboxesArrXYXY"
]


def bboxesArrXYXY_to_bboxesArrXYWH(
    bboxes_xyxy: BBoxesArrXYXYType
) -> BBoxesArrXYWHType:
    bboxes_xywh = bboxes_xyxy.copy()
    bboxes_xywh[:, [2, 3]] = bboxes_xyxy[:, [2, 3]] - bboxes_xyxy[:, [0, 1]]
    return bboxes_xywh

def bboxesArrXYWH_to_bboxesArrXYXY(
    bboxes_xywh: BBoxesArrXYWHType
) -> BBoxesArrXYXYType:
    bboxes_xyxy = bboxes_xywh.copy()
    bboxes_xyxy[:, [2, 3]] = bboxes_xywh[:, [0, 1]] + bboxes_xywh[:, [2, 3]]
    return bboxes_xyxy

def bboxesArrXYXY_to_bboxesLabelme(
    bboxes_xyxy: BBoxesArrXYXYType
) -> BBoxesLabelmeType:
    bboxes_labelme = bboxes_xyxy.copy().reshape(-1, 2, 2).tolist()
    return bboxes_labelme

def bboxesArrXYXY_to_bboxesCoco(
    bboxes: BBoxesArrXYXYType
) -> BBoxesCocoType:
    bboxes = bboxes.copy()
    bboxes[:, [2, 3]] = bboxes[:, [2, 3]] - bboxes[:, [0, 1]]
    bboxes = bboxes.tolist()
    return bboxes

def bboxesArrXYXY_to_bboxesYolo(
    bboxes: BBoxesArrXYXYType,
    img_hw: Tuple[int, int]
) -> BBoxesYoloType:
    img_h, img_w = img_hw
    bboxes = bboxes.copy()
    bboxes[:, [2, 3]] = bboxes[:, [2, 3]] - bboxes[:, [0, 1]]
    bboxes[:, [0, 1]] = bboxes[:, [0, 1]] + 0.5 * bboxes[:, [2, 3]]
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / img_w
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / img_h
    bboxes = bboxes.tolist()
    return bboxes

def bboxesLabelme_to_bboxesArrXYXY(
    bboxes_labelme: BBoxesLabelmeType
) -> BBoxesArrXYXYType:
    bboxes_arr = np.asarray(bboxes_labelme).reshape(-1, 4)
    return bboxes_arr

def bboxesCoco_to_bboxesArrXYXY(
    bboxes_coco: BBoxesCocoType
) -> BBoxesArrXYXYType:
    bboxes = np.asarray(bboxes_coco).astype(np.int32)
    bboxes[:, [2, 3]] = bboxes[:, [0, 1]] + bboxes[:, [2, 3]]
    return bboxes

def bboxesYolo_to_bboxesArrXYXY(
    bboxes: BBoxesYoloType,
    img_hw: Tuple[int, int]
) -> BBoxesArrXYXYType:
    img_h, img_w = img_hw
    bboxes = np.asarray(bboxes)
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * img_w
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * img_h
    bboxes[:, [0, 1]] = bboxes[:, [0, 1]] - 0.5 * bboxes[:, [2, 3]]
    bboxes[:, [2, 3]] = bboxes[:, [0, 1]] + bboxes[:, [2, 3]]
    bboxes = bboxes.astype(np.int32)
    return bboxes