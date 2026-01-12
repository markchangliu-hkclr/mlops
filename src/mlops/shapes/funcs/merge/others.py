from typing import List

import numpy as np
import pycocotools.mask as pycocomask

from mlops.shapes.typing.masks import MaskArrType
from mlops.shapes.typing.rles import RleType


__all__ = [
    "merge_maskArrs",
    "merge_rles"
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