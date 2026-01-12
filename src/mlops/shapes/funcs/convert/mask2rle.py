from typing import List

import numpy as np
import pycocotools.mask as pycocomask

from mlops.shapes.typing.masks import MaskArrType
from mlops.shapes.typing.rles import RleType


__all__ = [
    "maskArr_to_rle",
    "maskArrs_to_rles"
]


def maskArr_to_rle(
    mask: MaskArrType
) -> RleType:
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = pycocomask.encode(mask)
    return rle

def maskArrs_to_rles(
    masks: List[MaskArrType]
) -> List[RleType]:
    masks = np.stack(masks, axis = 2)
    masks = np.asfortranarray(masks.astype(np.uint8))
    rles = pycocomask.encode(masks)
    return rles
