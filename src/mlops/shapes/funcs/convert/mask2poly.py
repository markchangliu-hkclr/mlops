
from mlops.shapes.funcs.cnt2poly import cnt2poly_labelme
from mlops.shapes.funcs.mask2cnt import mask2cnt_merge
from mlops.shapes.typing.mask import MaskType
from mlops.shapes.typing.poly import PolyLabelmeType


__all__ = [
    "mask2poly_labelme"
]


def mask2poly_labelme(
    mask: MaskType,
    approx_flag: bool
) -> PolyLabelmeType:
    cnt = mask2cnt_merge(mask, approx_flag)
    poly = cnt2poly_labelme(cnt)
    return poly