from abc import ABC, abstractmethod
from typing import Any, List, Literal, Tuple

import numpy as np
from numpy.typing import NDArray

from mlops.labels.typedef.labelme import LabelmeDictType


class ImgPreProcessorABC(ABC):
    @property
    @abstractmethod
    def output_type(self) -> Literal["single", "multi"]:
        raise NotImplementedError

    def process_single_output(
        self,
        bgr: NDArray[np.uint8],
        *args: Any,
        **kwargs: Any
    ) -> NDArray[np.uint8]:
        raise NotImplementedError
    
    def process_multi_outputs(
        self,
        bgr: NDArray[np.uint8],
        *args: Any,
        **kwargs: Any
    ) -> List[NDArray[np.uint8]]:
        raise NotImplementedError

class LabelmePreprocessorABC(ABC):
    @property
    @abstractmethod
    def output_type(self) -> Literal["single", "multi"]:
        raise NotImplementedError

    def process_single_output(
        self,
        labelme_dict: LabelmeDictType,
        *args: Any,
        **kwargs: Any
    ) -> LabelmeDictType:
        raise NotImplementedError

    def process_multi_outputs(
        self,
        labelme_dict: LabelmeDictType,
        *args: Any,
        **kwargs: Any
    ) -> List[LabelmeDictType]:
        raise NotImplementedError

class DataPreprocessorABC(ABC):
    @property
    @abstractmethod
    def output_type(self) -> Literal["single", "multi"]:
        raise NotImplementedError

    def process_single_output(
        self,
        bgr: NDArray[np.uint8],
        labelme_dict: LabelmeDictType,
        *args: Any,
        **kwargs: Any
    ) -> Tuple[NDArray[np.uint8], LabelmeDictType]:
        raise NotImplementedError

    def process_multi_outputs(
        self,
        bgr: NDArray[np.uint8],
        labelme_dict: LabelmeDictType,
        *args: Any,
        **kwargs: Any
    ) -> Tuple[List[NDArray[np.uint8]], List[LabelmeDictType]]:
        raise NotImplementedError
