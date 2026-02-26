from typing import Tuple, List

import numpy as np
import pycocotools.mask as pycocomask

from mlops.shapes.typing.masks import MaskArrType, ContourType
from mlops.shapes.funcs.convert.contour2rle import *


__all__ = [
    "contour_to_maskArr",
    "contours_to_maskArrs",
]


def contour_to_maskArr(
    contour: ContourType,
    img_hw: Tuple[int, int]
) -> MaskArrType:
    rle = contour_to_rle(contour, img_hw)
    mask = pycocomask.decode(rle).astype(np.bool_)
    return mask

def contours_to_maskArrs(
    contours: List[ContourType],
    img_hw: Tuple[int, int],
    merge_flag: bool
) -> List[MaskArrType]:
    rles = contours_to_rles(contours, img_hw, merge_flag)
    masks = [pycocomask.decode(r).astype(np.bool_) for r in rles]
    return masks