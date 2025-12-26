from typing import Union, List, Any, Dict

import numpy as np
from numpy.typing import NDArray, DTypeLike


__all__ = ["BaseCategoryIDs", "BaseConfidences"]


class ArrayFormat:
    def __init__(
        self,
        dim: int,
        shape_dict: Dict[int, int],
        dtype: DTypeLike,
        strict_dtype_flag: bool
    ) -> None:
        self.dim = dim
        self.shape_dict = shape_dict
        self.dtype = dtype
        self.strict_dtype_flag = strict_dtype_flag
    
    def check(
        self, 
        array: NDArray[Any],
    ) -> None:
        assert len(array.shape) == self.dim
        assert array.dtype == self.dtype
        
        for d, s in self.shape_dict.items():
            assert array.shape[d] == s

        if self.strict_dtype_flag:
            assert array.dtype == self.dtype

class FormattedArray:
    """
    `(N, ), np array`
    """
    def __init__(
        self,
        data: NDArray[Any],
        dim: int,
        shape_dict: Dict[int, int],
        dtype: DTypeLike,
        strict_dtype_flag: bool
    ) -> None:
        self.format = ArrayFormat(dim, shape_dict, dtype, strict_dtype_flag)
        self.format.check(data)
        self._data = data
    
    def __getitem__(
        self, 
        item: Union[int, List[int], slice, NDArray[np.integer], NDArray[np.bool]]
    ) -> "FormattedArray":
        new_data = self._data[item]
        self._init_lazy(new_data)
        return self
    
    def __str__(self) -> str:
        return str(self._data)
    
    def update(self, data: NDArray[Any]) -> None:
        self.format.check(data)
        self._data = data
    
    def update_lazy(self, data: NDArray[Any]) -> None:
        self._data = data
    

    
    def __getitem__(
        self, 
        item: Union[int, List[int], slice, NDArray[np.integer], NDArray[np.bool]]
    ) -> "BaseCategoryIDs":
        new_data = self._data[item]
        self._init_lazy(new_data)
        return self
    
    def _init_lazy(
        self, 
        data: NDArray[np.integer]
    ) -> None:
        self._data = data


class BaseCategoryIDs(Array1D):
    """
    `(N, ), int32, np array`
    """
    def __init__(
        self,
        data: NDArray[np.integer]
    ) -> None:
        assert len(data.shape) == 1
        self._data = data.astype(np.int32)
    
    

class BaseConfidences:
    """
    `(N, ), float32, np array`
    """
    def __init__(
        self,
        data: NDArray[np.number]
    ) -> None:
        assert len(data.shape) == 1
        self._data = data.astype(np.float32)