import random
from typing import Tuple, List, Literal

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
        patch_hw: Tuple[int, int],
        edge_pad_include_prob: float
    ) -> None:
        self.pad_val = pad_val
        self.patch_hw = patch_hw
        self.edge_pad_include_prob = edge_pad_include_prob
        self._output_type = "multi"
    
    @property
    def output_type(self) -> Literal["single", "multi"]:
        return self._output_type
    
    def process_multi_outputs(
        self, 
        bgr: NDArray[np.uint8], 
        labelme_dict: LabelmeDictType
    ) -> Tuple[List[NDArray[np.uint8]], List[LabelmeDictType]]:
        bgr_patches_ = crop_img2patches(
            bgr, self.pad_val, self.patch_hw
        )
        labelme_patches_ = crop_labelme2patches(
            labelme_dict, self.patch_hw
        )

        assert len(bgr_patches_) == len(labelme_patches_)

        bgr_patches = []
        labelme_patches = []

        for b_patch, l_patch in zip(bgr_patches_, labelme_patches_):
            b_patch_channel_avg = np.mean(b_patch, axis = 2).astype(np.uint8)
            pad_area = np.sum(b_patch_channel_avg == self.pad_val)
            pad_area_ratio = pad_area / b_patch_channel_avg.size

            dice = random.uniform(0, 1.0)

            if pad_area_ratio < 0.6:
                bgr_patches.append(b_patch)
                labelme_patches.append(l_patch)
            elif self.edge_pad_include_prob > dice:
                bgr_patches.append(b_patch)
                labelme_patches.append(l_patch)
            else:
                continue

        return bgr_patches, labelme_patches