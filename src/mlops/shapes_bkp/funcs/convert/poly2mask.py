from typing import Tuple, List

from mlops.shapes.funcs.convert.poly2rle import \
    polyLabelme_to_rle, polysCoco_to_rles, polysYolo_to_rles
from mlops.shapes.funcs.convert.rle2mask import rle_to_mask
from mlops.shapes.typing.polys import PolyLabelmeType, PolysCocoType, PolysYoloType
from mlops.shapes.typing.masks import MaskArrType


__all__ = [
    "polyLabelme_to_maskArr",
    "polysCoco_to_maskArr",
]


def polyLabelme_to_maskArr(
    poly: PolyLabelmeType,
    img_hw: Tuple[int, int]
) -> MaskArrType:
    rle = polyLabelme_to_rle(poly, img_hw)
    mask = rle_to_mask(rle)
    return mask

def polysCoco_to_masksArr(
    polys: PolysCocoType,
    img_hw: Tuple[int, int],
    merge_flag: bool
) -> List[MaskArrType]:
    rles = polysCoco_to_rles(polys, img_hw, merge_flag)
    masks = [rle_to_mask(r) for r in rles]
    return masks

def polysYolo_to_maskArr(
    polys: PolysYoloType,
    img_hw: Tuple[int, int],
    merge_flag: bool
) -> List[MaskArrType]:
    rles = polysCoco_to_rles(polys, img_hw, merge_flag)
    masks = [rle_to_mask(r) for r in rles]
    return masks