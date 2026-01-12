import numpy as np

from mlops.shapes.typing.masks import MaskArrType


def merge_masks(
    masks: MaskArrType
) -> MaskArrType:
    masks = np.stack(masks, axis = 0) * 1
    mask_merge = np.sum(masks, axis = 0).astype(np.bool_)
    return mask_merge