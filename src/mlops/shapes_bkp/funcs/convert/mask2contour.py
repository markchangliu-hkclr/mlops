from typing import List

import numpy as np
import cv2

from mlops.shapes.typing.masks import MaskArrType
from mlops.shapes.typing.contours import ContourType
from mlops.shapes.funcs.merge.contours import merge_contours


__all__ = [
    "maskArr_to_contour",
]


def maskArr_to_contour(
    mask: MaskArrType,
    approx_flag: bool,
    merge_flag: bool
) -> List[ContourType]:
    mask_img = mask.astype(np.uint8) * 255
    cnts, hierarchy = cv2.findContours(mask_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

    if not approx_flag:
        cnts_approx = cnts
    else:
        cnts_approx = []
        for cnt in cnts:
            eps = 0.001 * cv2.arcLength(cnt, True)
            cnt_approx = cv2.approxPolyDP(cnt, eps, True)
            cnts_approx.append(cnt_approx)
    
    if not merge_flag:
        cnts_output = cnts_approx
    else:
        cnt_merge = merge_contours(cnts_approx, False, hierarchy)
        cnts_output = [cnt_merge]
    
    return cnts_output
