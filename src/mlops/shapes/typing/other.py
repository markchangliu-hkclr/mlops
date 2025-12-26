
try:
    from typing import TypeAlias
except:
    from typing_extensions import TypeAlias

import numpy as np
import numpy.typing as npt


__all__ = [
    "CatIDsType", 
    "ConfsType"
]


CatIDsType: TypeAlias = npt.NDArray[np.integer]
"""
`CatIDsType`
    `NDArray[np.integer]`, `(num_cat_ids, )`
"""

ConfsType: TypeAlias = npt.NDArray[np.floating]
"""
`ConfsType`
    `NDArray[np.floating]`, `(num_confs, )`
"""