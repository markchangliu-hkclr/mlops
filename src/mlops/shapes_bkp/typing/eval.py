from typing import TypeAlias, Union

import numpy as np
from numpy.typing import NDArray


__all__ = [
    "PredMatchRes1DType",
    "PredMatchRes2DType",
    "PredMatchResType",
    "GTMatchRes1DType",
    "GTMatchRes2DType"
    "GTMatchResType"
]


PredMatchRes1DType: TypeAlias = NDArray[np.integer]
"""
`NDArray[np.integer], (num_preds, ), matched gt ids, -1 means no match`
"""

PredMatchRes2DType: TypeAlias = NDArray[np.integer]
"""
`NDArray[np.bool_], (num_preds, num_gts), matched gt flags`
"""

PredMatchResType: TypeAlias = Union[PredMatchRes1DType, PredMatchRes2DType]
"""
- `NDArray[np.integer], (num_preds, ), matched gt ids, -1 means no match`
- `NDArray[np.bool_], (num_preds, num_gts), matched gt flags`
"""

GTMatchRes1DType: TypeAlias = NDArray[np.integer]
"""
`NDArray[np.integer], (num_gts, ), matched gt ids, -1 means no match`
"""

GTMatchRes2DType: TypeAlias = NDArray[np.integer]
"""
`NDArray[np.bool_], (num_gts, num_preds), matched gt flags`
"""

GTMatchResType: TypeAlias = Union[GTMatchRes1DType, GTMatchRes2DType]
"""
- `NDArray[np.integer], (num_gts, ), matched pred ids, -1 means no match`
- `NDArray[np.bool_], (num_gts, num_preds), matched pred flags`
"""