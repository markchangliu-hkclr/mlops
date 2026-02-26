from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


__all__ = [
    "ConfidencesArrType",
    "CategoryIDsArrType"
]


ConfidencesArrType: TypeAlias = NDArray[np.floating]
"""
`(n, ), float, np array`
"""

CategoryIDsArrType: TypeAlias = NDArray[np.floating]
"""
`(n, ), int, np array`
"""