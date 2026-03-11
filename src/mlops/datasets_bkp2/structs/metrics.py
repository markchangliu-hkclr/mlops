from dataclasses import dataclass
from typing import Dict

import numpy as np
from numpy.typing import NDArray


__all__ = [
    "PrecRecTable",
    "MAPTable"
]


@dataclass
class PrecRecTable:
    """
    Attrs
    -----
    - `confs: NDArray[np.floating], (num_preds, )`
    - `cum_tps: NDArray[np.integer], (num_preds, )`
    - `precs: NDArray[np.floating], (num_preds, )`
    - `precs_smooth: NDArray[np.floating], (num_preds, )`
    - `recs: NDArray[np.floating], (num_preds, )`
    - `recs_smooth: NDArray[np.floating], (num_preds, )`
    """
    confs: NDArray[np.floating]
    cum_tps: NDArray[np.integer]
    precs: NDArray[np.floating]
    precs_smooth: NDArray[np.floating]
    recs: NDArray[np.floating]
    recs_smooth: NDArray[np.floating]

@dataclass
class MAPTable:
    """
    Attrs
    -----
    - `aps: Dict[int, float], {cat_id, ap}`
    - `map: float`
    """
    aps: Dict[int, float]
    map: float

@dataclass
class ConfAccTable:
    """
    Indicate whether a model is well-calibrated.
    Well-calibrated means a high confidence prediction is usually a TP.
    """
    