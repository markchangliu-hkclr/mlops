from typing import List

import pycocotools.mask as pycocomask

from mlops.shapes.typing.rles import RleType


def merge_rles(
    rles: List[RleType]
) -> RleType:
    rle_merge = pycocomask.merge(rles, intersect = 0)
    return rle_merge