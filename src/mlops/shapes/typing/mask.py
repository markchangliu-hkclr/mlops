from typing import Tuple, TypeAlias

import numpy as np
import numpy.typing as npt


__all__ = [
    "MaskArrType",
    "MaskImgType",

    "MasksArrType",
    
    "MaskShapeType",
    "MasksShapeType"
]


MaskArrType: TypeAlias = npt.NDArray[np.bool_]
"""
`MaskArrType`
    `NDArray[np.bool_]`, `(img_h, img_w)`
"""

MaskShapeType: TypeAlias = Tuple[int, int]
"""
`MaskShapeType`
    `Tuple[int, int]`, `[img_h, img_w]`
"""

MaskImgType: TypeAlias = npt.NDArray[np.uint8]
"""
`MaskImgType`
    `NDArray[np.uint8]`, `(img_h, img_w)`, 0 ~ 255
"""

MasksArrType: TypeAlias = npt.NDArray[np.bool_]
"""
`MasksArrType`
    `NDArray[np.bool_]`, `(num_masks, img_h, img_w)`
"""

MasksShapeType: TypeAlias = Tuple[int, int, int]
"""
`MasksShapeType`
    `Tuple[int, int, int]`, `[num_masks, img_h, img_w]`
"""