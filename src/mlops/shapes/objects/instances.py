from typing import Optional, Union, List

import numpy as np
from numpy.typing import NDArray

from .bboxes import BaseBBoxes
from .masks import BaseMasks
from .array1d import BaseCategoryIDs, BaseConfidences


__all__ = ["Instances"]


class BaseInstances:
    def __init__(
        self,
        confs: BaseConfidences,
        cat_ids: BaseCategoryIDs,
        bboxes: BaseBBoxes,
        masks: Optional[BaseMasks],
    ) -> None:
        assert len(confs) == len(cat_ids) == len(bboxes)

        if masks is not None:
            assert len(confs) == len(masks)
        
        self.confs = confs.astype(np.float32)
        self.cat_ids = cat_ids.astype(np.int32)
        self.bboxes = bboxes.astype(np.int32)

        if masks is not None:
            self.masks = masks.astype(np.bool_)
        else:
            self.masks = None
    
    def __len__(self) -> int:
        return len(self.confs)

    def __getitem__(
        self, 
        item: Union[int, List[int], slice, npt.NDArray[np.integer]]
    ) -> "Insts":
        if isinstance(item, int):
            item = [item]
        
        new_confs = self.confs[item]
        new_cat_ids = self.cat_ids[item]
        new_bboxes = self.bboxes[item, :]

        if self.masks is not None:
            new_masks = self.masks[item, :]
        else:
            new_masks = None

        new_insts = Insts(
            new_confs, 
            new_cat_ids, 
            new_bboxes, 
            new_masks
        )

        return new_insts
    
    def sort_by_conf(self) -> None:
        sort_indice = np.argsort(self.confs)[::-1]

        self.confs = self.confs[sort_indice]
        self.cat_ids = self.confs[sort_indice]
        self.bboxes = self.bboxes[sort_indice, ...]
        
        if self.masks is not None:
            self.masks = self.masks[sort_indice, ...]
    
    def concat(
        self, 
        other_insts_list: List["Insts"]
    ) -> None:
        confs_list = [self.confs]
        cat_ids_list = [self.cat_ids]
        bboxes_list = [self.bboxes]
        masks_list = [self.masks]

        for insts in other_insts_list:
            confs_list.append(insts.confs)
            cat_ids_list.append(insts.cat_ids)
            bboxes_list.append(insts.bboxes)
            masks_list.append(insts.masks)
        
        self.confs = np.concat(confs_list)
        self.cat_ids = np.concat(cat_ids_list)
        self.bboxes = np.concat(bboxes_list, axis = 0)

        if self.masks is not None:
            self.masks = np.concat(masks_list, axis = 0)
    
    def filter_by_conf(
        self, 
        conf_thres: float,
        topk: int,
        sorted_flag: bool
    ) -> None:
        if not sorted_flag:
            self.sort_by_conf()
        
        filtered_flags = self.confs > conf_thres

        self.confs = self.confs[filtered_flags][:topk]
        self.cat_ids = self.cat_ids[filtered_flags][:topk]
        self.bboxes = self.bboxes[filtered_flags, ...][:topk, ...]

        if not self.masks is None:
            self.masks = self.masks[filtered_flags, ...][:topk, ...]