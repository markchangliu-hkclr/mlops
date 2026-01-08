import numpy as np
import pycocotools.mask as pycocomask

import mlops.shapes.typing.mask as mask_type
import mlops.shapes.typing.rle as rle_type


__all__ = [
    "mask2rle",
    "masks2rles"
]


def mask2rle(
    mask: mask_type.MaskType
) -> rle_type.RLEType:
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = pycocomask.encode(mask)
    return rle

def masks2rles(
    masks: mask_type.MasksType
) -> rle_type.RLEsType:
    masks = np.transpose(masks, (1, 2, 0))
    masks = np.asfortranarray(masks.astype(np.uint8))
    rles = pycocomask.encode(masks)
    return rles