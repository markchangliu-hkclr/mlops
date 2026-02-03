from typing import List

import numpy as np
import pycocotools.mask as pycocomask

from mlops.shapes.typing.bboxes import BBoxArrXYXYType, BBoxesArrXYXYType
from mlops.shapes.typing.masks import MaskArrType, MasksArrType
from mlops.shapes.typing.rles import RleType


__all__ = [
    "merge_maskArrs",
    "merge_rles",
    "merge_bboxArrsXYXY",
    "merge_masksArr"
]


def merge_maskArrs(
    masks: MaskArrType
) -> MaskArrType:
    masks = np.stack(masks, axis = 0) * 1
    mask_merge = np.sum(masks, axis = 0).astype(np.bool_)
    return mask_merge

def merge_rles(
    rles: List[RleType]
) -> RleType:
    rle_merge = pycocomask.merge(rles, intersect = 0)
    return rle_merge

def merge_bboxArrsXYXY(
    bboxes: List[BBoxArrXYXYType]
) -> BBoxArrXYXYType:
    bboxes = np.stack(bboxes, axis = 0)
    x1_merge = np.max(bboxes[:, 0]).item()
    y1_merge = np.max(bboxes[:, 1]).item()
    x2_merge = np.max(bboxes[:, 2]).item()
    y2_merge = np.max(bboxes[:, 3]).item()
    bbox_merge = np.asarray([x1_merge, y1_merge, x2_merge, y2_merge])
    return bbox_merge

def merge_bboxesArrXYXY(
    bboxes: BBoxesArrXYXYType
) -> BBoxArrXYXYType:
    x1_merge = np.max(bboxes[:, 0]).item()
    y1_merge = np.max(bboxes[:, 1]).item()
    x2_merge = np.max(bboxes[:, 2]).item()
    y2_merge = np.max(bboxes[:, 3]).item()
    bbox_merge = np.asarray([x1_merge, y1_merge, x2_merge, y2_merge])
    return bbox_merge

def merge_masksArr(
    masks: MasksArrType
) -> MaskArrType:
    mask_merge = np.sum(masks, axis = 0).astype(np.bool_)
    return mask_merge