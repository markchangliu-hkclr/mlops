import pycocotools.mask as pycocomask

import numpy as np

from mlops.shapes.typing.masks import MaskArrType
from mlops.shapes.typing.rles import RleType


def rle_to_mask(
    rle: RleType
) -> MaskArrType:
    mask = pycocomask.decode([rle]).astype(np.bool_)[0]
    return mask