import math
import random
from typing import Tuple, List, Union

import numpy as np
from numpy.typing import NDArray


def crop_img(
    bgr: NDArray[np.uint8],
    x1y1x2y2: Tuple[int, int, int, int]
) -> NDArray[np.uint8]:
    org_h, org_w = bgr.shape[:2]
    x1, y1, x2, y2 = x1y1x2y2

    crop_y1 = y1
    crop_y2 = min(y2, org_h)
    crop_x1 = x1
    crop_x2 = min(x2, org_w)

    bgr = bgr[crop_y1:crop_y2, crop_x1:crop_x2, :]
    return bgr

def crop_img2patches(
    bgr: NDArray[np.uint8],
    pad_val: int,
    patch_hw: Tuple[int, int],
) -> List[NDArray[np.uint8]]:
    org_h, org_w = bgr.shape[:2]
    patch_h, patch_w = patch_hw

    patches = []

    num_step_h = math.ceil(org_h / patch_h)
    num_step_w = math.ceil(org_w / patch_w)

    if num_step_h == 1 and num_step_w == 1:
        patches.append(bgr)
        return patches

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
    
    x1_center_patch = max(org_w // 2 - patch_w // 2, 0)
    x2_center_patch = min(x1_center_patch + patch_w, org_w)
    y1_center_patch = max(org_h // 2 - patch_h // 2, 0)
    y2_center_patch = min(y1_center_patch + patch_h, org_h)
    center_patch = crop_img(
        bgr, (x1_center_patch, y1_center_patch, x2_center_patch, y2_center_patch)
    )

    patches.append(center_patch)

    return patches