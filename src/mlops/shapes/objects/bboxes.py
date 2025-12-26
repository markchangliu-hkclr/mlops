import numpy as np
from numpy.typing import NDArray


__all__ = ["BaseBBoxes"]



class BaseBBoxes:
    """
    `(N, 4), xyxy format, int32, np array`
    """
    def __init__(
        self, 
        data: NDArray[np.integer],
    ) -> None:
        assert len(data.shape) == 2
        assert data.shape[1] == 4
        assert np.all(data > 0).item()
        assert np.all(data[:, [2, 3]] > data[:, [0, 1]]).item()

        self._data = data.astype(np.int32)