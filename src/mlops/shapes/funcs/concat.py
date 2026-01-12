from typing import List, Union

import numpy as np

from mlops.shapes.typing.bboxes import *
from mlops.shapes.typing.masks import *
from mlops.shapes.typing.others import *


__all__ = [
    "concat_bboxArrXYXYs",
    "concat_maskArrs"
]


def concat_bboxArrXYXYs(
    bboxes_list: List[Union[BBoxesArrXYXYType, BBoxArrXYXYType]]
) -> BBoxesArrXYXYType:
    bboxes_list_ = []
    for b in bboxes_list:
        if len(b.shape) == 1:
            bboxes_list_.append(b[None, ...])
        elif len(b.shape) == 2:
            bboxes_list_.append(b)
        else:
            raise NotImplementedError

    bboxes_list = bboxes_list_
    bboxes_output = np.concat(bboxes_list, axis = 0).astype(np.int32)
    return bboxes_output

def concat_maskArrs(
    masks_list: List[Union[MasksArrType, MaskArrType]]
) -> MasksArrType:
    masks_list_ = []
    for m in masks_list:
        if len(m.shape) == 1:
            masks_list_.append(m[None, ...])
        elif len(m.shape) == 2:
            masks_list_.append(m)
        else:
            raise NotImplementedError

    masks_list = masks_list_
    masks_output = np.concat(masks_list, axis = 0).astype(np.bool_)
    return masks_output

def concat_confidenceArrs(
    confs_list: List[ConfidencesArrType]
) -> ConfidencesArrType:
    confs_output = np.concat(confs_list).astype(np.float32)
    return confs_output

def concat_categoryArrs(
    cats_list: List[CategoryIDsArrType]
) -> CategoryIDsArrType:
    cats_output = np.concat(cats_list).astype(np.int32)
    return cats_output