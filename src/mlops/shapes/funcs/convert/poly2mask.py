from typing import Tuple

from mlops.shapes.funcs.convert.poly2rle import \
    polyLabelmes_to_rles, polyLabelme_to_rle
from mlops.shapes.funcs.convert.rle2mask import rle_to_mask
from mlops.shapes.typing.polys import PolyLabelmeType
from mlops.shapes.typing.masks import MaskArrType


__all__ = [
    "polyLabelme_to_maskArr",
]


def polyLabelme_to_maskArr(
    poly: PolyLabelmeType,
    img_hw: Tuple[int, int]
) -> MaskArrType:
    rle = polyLabelme_to_rle(poly, img_hw)
    mask = rle_to_mask(rle)
    return mask