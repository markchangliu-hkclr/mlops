from types import MappingProxyType
from typing import Union, List

import numpy as np
from numpy.typing import NDArray

from ..typing import PolysArrType


class Polys:
    """
    `(N_polys, (N_points, 2)), int32, List[np array]`
    """
    spec = MappingProxyType({
        "shape": "(N_polys, (N_points, 2))",
        "dtype": "int32",
    })

    def __init__(
        self,
        data: PolysArrType,
        check_flag: bool
    ) -> None:
        if check_flag:
            self.check(data)
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)

    def check(
        self,
        data: PolysArrType
    ) -> None:
        for p in data:
            assert len(p.shape) == 2
            assert p.shape[1] == 2
    
    def getitem(
        self,
        item: Union[int, List[int], NDArray[np.integer], NDArray[np.bool_]],
        update_flag: bool
    ) -> "Polys":
        if isinstance(item, int):
            item = [item]
        elif isinstance(item, np.ndarray) and item.dtype == np.bool_:
            item = np.arange(len(self.data))[item].tolist()
        elif isinstance(item, np.ndarray):
            item = item.astype(np.int32).tolist()
        elif isinstance(item, list):
            pass
        else:
            raise NotImplementedError
        
        new_data = [self.data[i] for i in item]

        if update_flag:
            self.data = new_data
            return self
        else:
            new_polys = Polys(new_data, False)
            return new_polys
    
    def sort(
        self,
        sort_key: Union[List[int], NDArray[np.integer]],
        update_flag: bool
    ) -> "Polys":
        assert len(sort_key) == len(self.data)
        return self.getitem(sort_key, update_flag)
    
    def filter(
        self,
        filter_key: Union[List[bool], NDArray[np.bool_]],
        update_flag: bool
    ) -> "Polys":
        assert len(filter_key) == len(self.data)
        return self.getitem(filter_key, update_flag)
    
    def concat(
        self, 
        polys_list: List["Polys"],
        update_flag: bool
    ) -> "Polys":
        new_polys = self.data
        for p in polys_list:
            new_polys += p.data

        if update_flag:
            self.data = polys_list
            return self
        else:
            new_polys = Polys(new_polys, False)
            return new_polys