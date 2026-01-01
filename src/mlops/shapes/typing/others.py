from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


__all__ = [
    "ConfidencesArrtype",
    "CategoryIDsArrType"
]


ConfidencesArrtype: TypeAlias
"""
`(n, ), float, np array`
"""

CategoryIDsArrType: TypeAlias
"""
`(n, ), int, np array`
"""