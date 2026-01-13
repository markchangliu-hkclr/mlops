from typing import List, Union

import numpy as np

from mlops.shapes.typing.bboxes import *
from mlops.shapes.typing.masks import *
from mlops.shapes.typing.others import *
from mlops.shapes.structs.instances import Instances


__all__ = [
    "concat_bboxArrXYXYs",
    "concat_maskArrs",
    "concat_confidenceArrs",
    "concat_categoryArrs",
    "concat_instances"
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

def concat_instances(
    insts_list: List[Instances]
) -> Instances:
    confs_list = [i.confs for i in insts_list]
    cat_ids_list = [i.cat_ids for i in insts_list]
    bboxes_list = [i.bboxes for i in insts_list]
    masks_list = [i.masks for i in insts_list]

    new_confs = concat_confidenceArrs(confs_list)
    new_cat_ids = concat_categoryArrs(cat_ids_list)
    new_bboxes = concat_bboxArrXYXYs(bboxes_list)
    
    if None in masks_list:
        new_masks = None
    else:
        new_masks = concat_maskArrs(masks_list)
    
    new_insts = Instances(
        new_confs, new_cat_ids, new_bboxes, new_masks
    )

    return new_insts