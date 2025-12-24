from typing import Tuple, List

try:
    from typing import TypeAlias
except:
    from typing_extensions import TypeAlias

import numpy as np
import numpy.typing as npt


__all__ = [
    "BBoxXYXYType",
    "BBoxXYWHType",
    "BBoxLabelmeType",
    "BBoxesLabelmeType",
    "BBoxCocoType",
    "BBoxYoloType",

    "BBoxesXYXYType",
    "BBoxesXYWHType",
    "BBoxesLabelmeType",
]

BBoxXYXYType: TypeAlias = npt.NDArray[np.number]
"""
`BBoxXYXYType`
    `NDArray[np.number]`, `(4, )`, `[x1, y1, x2, y2]`
"""

BBoxXYWHType: TypeAlias = npt.NDArray[np.number]
"""
`BBoxXYWHType`
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

BBoxesXYXYType: TypeAlias = npt.NDArray[np.number]
"""
`BBoxesXYXYArrType`
    `NDArray[np.number]`, `(num_bboxes, 4)`, `[[x1, y1, w, h], ...]`
"""

BBoxesXYWHType: TypeAlias = npt.NDArray[np.number]
"""
`BBoxesXYWHArrType`
    `NDArray[np.number]`, `(num_bboxes, 4)`, `[[x1, y1, w, h], ...]`
"""

BBoxesLabelmeType: TypeAlias = List[BBoxLabelmeType]
"""
`BBoxesLabelmeType`
    `List[BBoxLabelmeType], `(n, (2, ))`, `[[[x1, y1], [x2, y2]], ...]`
"""
