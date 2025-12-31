from types import MappingProxyType
from typing import List, Union

import numpy as np
from numpy.typing import NDArray


__all__ = ["BBoxes"]



class BBoxes:
    """
    `(N, 4), xyxy, int32, np array`
    """
    spec = MappingProxyType({
        "shape": "(N, 4)",
        "format": "xyxy",
        "dtype": "int32"
    })

    def __init__(
        self, 
        data: NDArray[np.integer],
        check_flag: bool
    ) -> None:
        if check_flag:
            self.check(data)
        self.data = data.astype(np.int32)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(
        self,
        item: Union[int, List[int], NDArray[np.integer], NDArray[np.bool_]]
    ) -> "BBoxes":
        new_data = self.data[item, ...]
        new_bboxes = BBoxes(new_data, False)
        return new_bboxes
    
    def check(self, data: NDArray[np.integer]) -> None:
        assert len(data.shape) == 2
        assert data.shape[1] == 4
        assert np.all(data > 0).item()
        assert np.all(data[:, [2, 3]] > data[:, [0, 1]]).item()
    
    def concat(
        self, 
        bboxes_list: List["BBoxes"],
        update_flag: bool
    ) -> "BBoxes":
        bboxes_list = [self.data] + [b.data for b in bboxes_list]
        new_bboxes = np.concat(bboxes_list, axis = 0)

        if update_flag:
            self.data = new_bboxes.astype(np.int32)
            return self
        else:
            new_bboxes = BBoxes(new_bboxes, False)
            return new_bboxes