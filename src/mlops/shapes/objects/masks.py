from typing import Literal

import numpy as np
from numpy.typing import NDArray


__all__ = ["BaseMasks"]


class BaseMasks:
    """
    `(N, img_h, img_w), bool, np array`
    """
    def __init__(
        self, 
        data: NDArray[np.bool_],
    ) -> None:
        assert len(data.shape) == 3
        self._data = data