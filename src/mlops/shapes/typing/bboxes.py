from typing import Tuple, List, TypeAlias

import numpy as np
import numpy.typing as npt


__all__ = [
    "BBoxArrXYXYType",
    "BBoxArrXYWHType",
    "BBoxLabelmeType",
    "BBoxCocoType",
    "BBoxYoloType",

    "BBoxesArrXYXYType",
    "BBoxesArrXYWHType",
    "BBoxesLabelmeType",
]

BBoxArrXYXYType: TypeAlias = npt.NDArray[np.number]
"""
`BBoxArrXYXYType`
    `NDArray[np.number]`, `(4, )`, `[x1, y1, x2, y2]`
"""

BBoxArrXYWHType: TypeAlias = npt.NDArray[np.number]
"""
`BBoxArrXYWHType`
    `NDArray[np.number]`, `(4, )`, `[x1, y1, w, h]`
"""

BBoxLabelmeType: TypeAlias = Tuple[Tuple[float, float], Tuple[float, float]]
"""
`BBoxLabelmeType`
    `Tuple[Tuple[float, float], Tuple[float, float]], `(2, (2, ))`, `[[x1, y1], [x2, y2]]`
"""

BBoxCocoType: TypeAlias = Tuple[float, float, float, float]
"""
`BBoxCocoType`
    `Tuple[float, float, float, float]`, `(4, )`, `[x1, y1, w, h]`
"""

BBoxYoloType: TypeAlias = Tuple[float, float, float, float]
"""
`BBoxYoloType`
    `Tuple[float, float, float, float]`, `(4, )`, 
    `[x_ctr_norm, y_ctr_norm, w_norm, y_norm]`
"""

BBoxesArrXYXYType: TypeAlias = npt.NDArray[np.number]
"""
`BBoxesArrXYXYType`
    `NDArray[np.number]`, `(num_bboxes, 4)`, `[[x1, y1, w, h], ...]`
"""

BBoxesArrXYWHType: TypeAlias = npt.NDArray[np.number]
"""
`BBoxesArrXYWHType`
    `NDArray[np.number]`, `(num_bboxes, 4)`, `[[x1, y1, w, h], ...]`
"""

BBoxesLabelmeType: TypeAlias = List[BBoxLabelmeType]
"""
`BBoxesLabelmeType`
    `List[BBoxLabelmeType], `(n, (2, ))`, `[[[x1, y1], [x2, y2]], ...]`
"""
