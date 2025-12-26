from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from mlops.shapes.insts import Insts
from mlops.shapes.typedef.bboxes import BBoxesXYXYArrType
from mlops.shapes.typedef.masks import MasksType


class BaseModel(ABC):
    def infer(
        self, 
        bgr: npt.NDArray[np.uint8]
    ) -> Insts:
        raise NotImplementedError

    def infer_masks_given_bboxes(
        self,
        bgr: npt.NDArray[np.uint8],
        bboxes: BBoxesXYXYArrType
    ) -> MasksType:
        raise NotImplementedError

