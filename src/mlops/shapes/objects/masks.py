from types import MappingProxyType
from typing import Union, List

import numpy as np
from numpy.typing import NDArray


__all__ = ["Masks"]


class Masks:
    """
    `(N, img_h, img_w), bool, np array`
    """

    spec = MappingProxyType({
        "shape": "(N, img_h, img_w)",
        "dtype": "bool"
    })
    
    def __init__(
        self, 
        data: NDArray[np.bool_],
        check_flag: bool
    ) -> None:
        if check_flag:
            self.check(data)
        self.data = data.astype(np.bool_)
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self,
        item: Union[int, List[int], NDArray[np.integer], NDArray[np.bool_]]
    ) -> "Masks":
        new_data = self.data[item, ...]
        new_masks = Masks(new_data, False)
        return new_masks
    
    def check(self, data: NDArray[np.bool_]) -> None:
        assert len(data.shape) == 3
    
    def concat(
        self, 
        masks_list: List["Masks"],
        update_flag: bool
    ) -> "Masks":
        masks_list = [self.data] + [m.data for m in masks_list]
        new_masks = np.concat(masks_list, axis = 0)

        if update_flag:
            self.data = new_masks.astype(np.bool_)
            return self
        else:
            new_masks = Masks(new_masks, False)
            return new_masks