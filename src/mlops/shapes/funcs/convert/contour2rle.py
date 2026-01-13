from typing import Tuple, List

import pycocotools.mask as pycocomask

from mlops.shapes.typing.contours import ContourType
from mlops.shapes.typing.rles import RleType


__all__ = [
    "contour_to_rle",
    "contours_to_rles",
]


def contour_to_rle(
    contour: ContourType,
    img_hw: Tuple[int, int]
) -> RleType:
    poly_coco = contour.flatten().tolist()

    if len(poly_coco) == 4:
        poly_coco += poly_coco[-2:]

    rle = pycocomask.frPyObjects([poly_coco], img_hw[0], img_hw[1])[0]

    return rle

def contours_to_rles(
    contours: List[ContourType],
    img_hw: Tuple[int, int],
    merge_flag: bool
) -> List[RleType]:
    rles = []
    for cnt in contours:
        rle = contour_to_rle(cnt, img_hw)
        rles.append(rle)
    
    if not merge_flag:
        return rles
    
    rles = [pycocomask.merge(rles)]
    return rles
    