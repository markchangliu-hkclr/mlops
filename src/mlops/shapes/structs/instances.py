from typing import Union, List
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mlops.shapes.typing.bboxes import BBoxesArrXYXYType
from mlops.shapes.typing.masks import MasksArrType
from mlops.shapes.typing.others import ConfidencesArrType, CategoryIDsArrType


__all__ = [
    "Instances"
]


@dataclass
class Instances:
    confs: ConfidencesArrType
    cat_ids: CategoryIDsArrType
    bboxes: BBoxesArrXYXYType
    masks: Union[MasksArrType, None]

    def __post_init__(self) -> None:
        assert len(self.confs) == len(self.cat_ids) == len(self.bboxes)

        if self.masks is not None:
            assert len(self.masks) == len(self.confs)
        
        self.confs = self.confs.astype(np.float32)
        self.cat_ids = self.cat_ids.astype(np.int32)
        self.bboxes = self.bboxes.astype(np.int32)
        self.masks = self.masks.astype(np.bool_)

    def __len__(self) -> int:
        return 
    
    def getitem(
        self,
        item: Union[int, List[int], NDArray[np.integer], NDArray[np.bool_]],
        update_flag: bool
    ) -> "Instances":
        new_confs = self.confs.getitem(item, update_flag)
        new_cat_ids = self.cat_ids.getitem(item, update_flag)
        new_bboxes = self.bboxes.getitem(item, update_flag)

        if self.masks is not None:
            new_masks = self.masks.getitem(item, update_flag)
        else:
            new_masks = None
        
        if update_flag:
            self.confs = new_confs
            self.cat_ids = new_cat_ids
            self.bboxes = new_bboxes
            self.masks = new_masks
            return self
        else:
            new_insts = Instances(
                new_confs, new_cat_ids, new_bboxes, new_masks, False
            )
            return new_insts
    
    def sort_by_confs(
        self,
        update_flag: bool
    ) -> "Instances":
        sort_key = np.argsort(self.confs)[::-1]
        new_insts = self.getitem(sort_key, update_flag)
        return new_insts
    
    def filter_by_conf_thres(
        self,
        conf_thres: float,
        update_flag: bool
    ) -> "Instances":
        filter_key = self.confs > conf_thres
        new_insts = self.getitem(filter_key, update_flag)
        return new_insts
    
    def filter_to_topk(
        self,
        topk: int,
        update_flag: bool
    ) -> "Instances":
        sort_key = np.argsort(self.confs)[::-1][:topk]
        new_insts = self.getitem(sort_key, update_flag)
        return new_insts
    
    def concat(
        self,
        others: List["Instances"],
        update_flag: bool
    ) -> "Instances":
        confs_list = [self.confs] + [i.confs for i in others] 
        cat_ids_list = [self.cat_ids] + [i.cat_ids for i in others]
        bboxes_list = [self.bboxes] + [i.bboxes for i in others]

        new_confs = np.concat(confs_list)
        new_cat_ids = np.concat(cat_ids_list)
        new_bboxes = np.concat(bboxes_list, axis = 0)

        if self.masks is not None:
            masks_list = [self.masks] + [i.masks for i in others]
            new_masks = np.concat(masks_list, axis = 0)
        else:
            new_masks = None

        if update_flag:
            self.confs = new_confs
            self.cat_ids = new_cat_ids
            self.bboxes = new_bboxes
            self.masks = new_masks
            return self
        else:
            new_insts = Instances(
                new_confs, new_cat_ids, new_bboxes, new_masks, False
            )
            return new_insts