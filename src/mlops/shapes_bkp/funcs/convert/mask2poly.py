from typing import List, Tuple

from mlops.shapes.typing.masks import MaskArrType
from mlops.shapes.typing.polys import *
from mlops.shapes.funcs.convert.mask2contour import maskArr_to_contour
from mlops.shapes.funcs.convert.contour2poly import *


__all__ = [
    "maskArr_to_polyLabelmes",
    "maskArr_to_polyCocos",
    "maskArr_to_polyYolos"
]


def maskArr_to_polyLabelmes(
    mask: MaskArrType,
    approx_flag: bool,
    merge_flag: bool
) -> List[PolyLabelmeType]:
    cnts = maskArr_to_contour(mask, approx_flag, merge_flag)
    polys = contours_to_polyLabelmes(cnts)
    return polys

def maskArr_to_polyCocos(
    mask: MaskArrType,
    approx_flag: bool,
    merge_flag: bool
) -> List[PolyCocoType]:
    cnts = maskArr_to_contour(mask, approx_flag, merge_flag)
    polys = contours_to_polyCocos(cnts)
    return polys

def maskArr_to_polyYolos(
    mask: MaskArrType,
    approx_flag: bool,
    merge_flag: bool,
    img_hw: Tuple[int, int]
) -> List[PolyYoloType]:
    cnts = maskArr_to_contour(mask, approx_flag, merge_flag)
    polys = contours_to_polyYolos(cnts, img_hw)
    return polys

