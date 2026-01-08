from types import MappingProxyType
from typing import Union, List

import numpy as np
from numpy.typing import NDArray

from ..typing import RleDictType


class Rles:
    """
    `(num_rles, ), List[dict]`
    """
    spec = MappingProxyType({
        "shape": "(N_rles, )",
        "format": "dict",
    })

    def __init__(
        self,
        data: List[RleDictType],
        check_flag: bool
    ) -> None:
        if check_flag:
            self.check(data)
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)

    def check(
        self, 
        data: List[RleDictType]
    ) -> None:
        for d in data:
            assert isinstance(d, dict)
            assert "size" in d.keys()
            assert "counts" in d.keys()
    
    def getitem(
        self,
        item: Union[int, List[int], NDArray[np.integer], NDArray[np.bool_]],
        update_flag: bool
    ) -> "Rles":
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
            new_rles = Rles(new_data, False)
            return new_rles
    
    def sort(
        self,
        sort_key: Union[List[int], NDArray[np.integer]],
        update_flag: bool
    ) -> "Rles":
        assert len(sort_key) == len(self.data)
        return self.getitem(sort_key, update_flag)
    
    def filter(
        self,
        filter_key: Union[List[bool], NDArray[np.bool_]],
        update_flag: bool
    ) -> "Rles":
        assert len(filter_key) == len(self.data)
        return self.getitem(filter_key, update_flag)
    
    def concat(
        self, 
        rles_list: List["Rles"],
        update_flag: bool
    ) -> "Rles":
        new_data = self.data
        for r in rles_list:
            new_polys += r.data

        if update_flag:
            self.data = new_data
            return self
        else:
            new_rles = Rles(new_polys, False)
            return new_rles
