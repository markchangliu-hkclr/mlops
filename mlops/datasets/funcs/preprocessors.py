from typing import Tuple, List

import numpy as np
from numpy.typing import NDArray

from mlops.datasets.core.abcs import DataPreprocessorABC
from mlops.datasets.funcs.ops.image import crop_img2patches
from mlops.datasets.funcs.ops.labelme import crop_labelme2patches
from mlops.labels.typedef.labelme import LabelmeDictType


class DataCropPatchPreprocessor(DataPreprocessorABC):
    def __init__(
        self,
        pad_val: int,
        patch_hw: Tuple[int, int]
    ) -> None:
        self.pad_val = pad_val
        self.patch_hw = patch_hw
        self.output_type = "multi"
    
    def process_multi_outputs(
        self, 
        bgr: NDArray[np.uint8], 
        labelme_dict: LabelmeDictType
    ) -> Tuple[List[NDArray[np.uint8]], List[LabelmeDictType]]:
        bgr_patches = crop_img2patches(
            bgr, self.pad_val, self.patch_hw
        )
        labelme_patches = crop_labelme2patches(
            labelme_dict, self.patch_hw
        )
        return bgr_patches, labelme_patches