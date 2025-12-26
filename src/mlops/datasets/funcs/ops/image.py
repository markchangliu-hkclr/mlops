import math
from typing import Tuple, List

import numpy as np
from numpy.typing import NDArray


def crop_img(
    bgr: NDArray[np.uint8],
    x1y1x2x2: Tuple[int, int, int, int]
) -> NDArray[np.uint8]:
    x1, y1, x2, y2 = x1y1x2x2
    bgr = bgr[y1:y2, x1:x2, :]
    return bgr

def crop_img2patches(
    bgr: NDArray[np.uint8],
    pad_val: int,
    patch_hw: Tuple[int, int],
) -> List[NDArray[np.uint8]]:
    org_h, org_w = bgr.shape[:2]
    patch_h, patch_w = patch_hw

    num_step_h = math.ceil(org_h / patch_h)
    num_step_w = math.ceil(org_w / patch_w)

    patches = []

    for i in range(num_step_h):
        for j in range(num_step_w):
            y1 = i * patch_h
            y2 = (i + 1) * patch_h
            x1 = j * patch_w
            x2 = (j + 1) * patch_w

            crop_y1 = y1
            crop_y2 = min(y2, org_h)
            crop_x1 = x1
            crop_x2 = min(x2, org_w)
            crop_h = crop_y2 - crop_y1
            crop_w = crop_x2 - crop_x1

            patch_bgr = np.ones((patch_h, patch_w, 3)).astype(np.uint8) * pad_val
            patch_bgr[0:crop_h, 0:crop_w, :] = bgr[crop_y1:crop_y2, crop_x1:crop_x2, :]
            
            patches.append(patch_bgr)

    return patches