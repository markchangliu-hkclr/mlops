from types import MappingProxyType
from typing import List, Literal, Union

import numpy as np
from numpy.typing import NDArray


__all__ = ["CategoryIDs", "Confidences"]


class CategoryIDs:
    """
    `(N, ), int32`
    """

    spec = MappingProxyType({
        "shape": "(N, )",
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
    
    def check(self, data: NDArray[np.integer]) -> None:
        assert len(data.shape) == 1
    
    def getitem(
        self,
        item: Union[int, List[int], NDArray[np.integer], NDArray[np.bool_]],
        update_flag: bool
    ) -> "CategoryIDs":
        new_data = self.data[item]

        if update_flag:
            self.data = new_data
            return self
        else:
            new_cat_ids = CategoryIDs(new_data, False)
            return new_cat_ids
    
    def sort(
        self,
        sort_key: Union[List[int], NDArray[np.integer]],
        update_flag: bool
    ) -> "CategoryIDs":
        return self.getitem(sort_key, update_flag)
    
    def filter(
        self,
        filter_key: Union[List[bool], NDArray[np.bool_]],
        update_flag: bool
    ) -> "CategoryIDs":
        return self.getitem(filter_key, update_flag)
    
    def concat(
        self,
        cat_ids_list: List["CategoryIDs"],
        update_flag: bool
    ) -> "CategoryIDs":
        new_data = [self.data] + [c.data for c in cat_ids_list]
        new_data = np.concat(new_data, axis = 0)

        if update_flag:
            self.data = new_data.astype(np.int32)
            return self
        else:
            new_cat_ids = CategoryIDs(new_data, False)
            return new_cat_ids


class Confidences:
    """
    `(N, ), float32`
    """

    spec = MappingProxyType({
        "shape": "(N, )",
        "dtype": "float32"
    })

    def __init__(
        self,
        data: NDArray[np.floating],
        check_flag: bool
    ) -> None:
        if check_flag:
            self.check(data)
        self.data = data.astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def getitem(
        self,
        item: Union[int, List[int], NDArray[np.integer], NDArray[np.bool_]],
        update_flag: bool
    ) -> "Confidences":
        new_data = self.data[item]

        if update_flag:
            self.data = new_data
            return self
        else:
            new_cat_ids = Confidences(new_data, False)
            return new_cat_ids
    
    def check(self, data: NDArray[np.floating]) -> None:
        assert len(data.shape) == 1
    
    def argsort(
        self,
        mode: Literal["ascend", "descend"]
    ) -> NDArray[np.integer]:
        sort_key = np.argsort(self.data)

        if mode == "descend":
            sort_key = sort_key[::-1]
        elif mode == "ascend":
            pass
        else:
            raise NotImplementedError
        
        return sort_key
    
    def argfilter(
        self,
        thres: float,
        mode: Literal["lower", "upper"]
    ) -> NDArray[np.bool_]:
        if mode == "lower":
            filter_key = self.data <= thres
        elif mode == "upper":
            filter_key = self.data >= thres
        else:
            raise NotImplementedError
        
        return filter_key
    
    def sort(
        self,
        sort_key: Union[List[int], NDArray[np.integer]],
        update_flag: bool
    ) -> "Confidences":
        assert len(sort_key) == len(self.data)
        return self.getitem(sort_key, update_flag)
    
    def filter(
        self,
        filter_key: Union[List[bool], NDArray[np.bool_]],
        update_flag: bool
    ) -> "Confidences":
        assert len(filter_key) == len(self.data)
        return self.getitem(filter_key, update_flag)
    
    def concat(
        self,
        confs_list: List["Confidences"],
        update_flag: bool
    ) -> "Confidences":
        new_data = [self.data] + [c.data for c in confs_list]
        new_data = np.concat(new_data, axis = 0)

        if update_flag:
            self.data = new_data.astype(np.float32)
            return self
        else:
            new_cat_ids = Confidences(new_data, False)
            return new_cat_ids