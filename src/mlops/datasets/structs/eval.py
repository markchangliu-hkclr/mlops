from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


__all__ = [
    "DetResult"
]


@dataclass
class DetResult:
    """
    Attrs
    -----
    - `confs: NDArray[np.floating], (num_preds, )`
    - `tp_pred_flags: NDArray[np.bool_], (num_preds, )`
    - `fn_gt_flags: NDArray[np.bool_], (num_gts, )`
    """
    confs: NDArray[np.floating]
    tp_pred_flags: NDArray[np.bool_]
    fn_gt_flags: NDArray[np.bool_]

@dataclass
class PrecRecTable:
    """
    Attrs
    -----
    - `confs: NDArray[np.floating], (num_preds, )`
    - `cum_tps: NDArray[np.integer], (num_preds, )`
    - `precs: NDArray[np.floating], (num_preds, )`
    - `recs: NDArray[np.floating], (num_preds, )`
    """
    confs: NDArray[np.floating]
    cum_tps: NDArray[np.integer]
    precs: NDArray[np.floating]
    recs: NDArray[np.floating]