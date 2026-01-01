from typing import Literal, Union, List

import numpy as np
from numpy.typing import NDArray

from .bboxes import BBoxes
from .masks import Masks
from .others import CategoryIDs, Confidences


__all__ = ["Instances"]


class Instances:
    """
    `confs, cat_ids, bboxes, Optional[masks]`
    """

    def __init__(
        self,
        confs: Confidences,
        cat_ids: CategoryIDs,
        bboxes: BBoxes,
        masks: Union[None, Masks],
        check_flag: bool
    ) -> None:
        if check_flag:
            self.check(confs, cat_ids, bboxes, masks)
        
        self.confs = confs
        self.cat_ids = cat_ids
        self.bboxes = bboxes
        self.masks = masks
    
    def __len__(self) -> int:
        return len(self.confs)

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
    
    def check(
        self, 
        confs: Confidences,
        cat_ids: CategoryIDs,
        bboxes: BBoxes,
        masks: Union[None, Masks],
    ) -> None:
        confs.check()
        cat_ids.check()
        bboxes.check()
        assert len(confs) == len(cat_ids) == len(bboxes)

        if masks is not None:
            masks.check()
            assert len(confs) == len(masks)

    def sort_by_conf(
        self,
        update_flag: bool
    ) -> "Instances":
        sort_key = self.confs.argsort("descend")
        new_confs = self.confs.sort(sort_key, update_flag)
        new_cat_ids = self.cat_ids.sort(sort_key, update_flag)
        new_bboxes = self.bboxes.sort(sort_key, update_flag)
        
        if self.masks is not None:
            new_masks = self.masks.sort(sort_key, update_flag)
        else:
            new_masks = None
        
        if update_flag:
            self.confs = new_confs
            self.cat_ids = new_cat_ids
            self.bboxes = new_bboxes
            self.masks = new_bboxes
            return self
        else:
            new_insts = Instances(
                new_confs, new_cat_ids, new_bboxes, new_masks, False
            )
            return new_insts
    
    def filter_by_conf(
        self,
        conf_thres: float,
        mode: Literal["lower", "upper"],
        update_flag: bool
    ) -> "Instances":
        filter_key = self.confs.argfilter(conf_thres, mode)
        new_insts = self.getitem(filter_key, update_flag)
        return new_insts
    
    def filter(
        self,
        filter_key: Union[List[bool], NDArray[np.bool_]],
        update_flag: bool
    ) -> "Instances":
        assert len(filter_key) == len(self)
        new_insts = self.getitem(filter_key, update_flag)
        return self
    
    def concat(
        self, 
        other_insts_list: List["Instances"],
        update_flag: bool
    ) -> "Instances":
        other_confs = [i.confs for i in other_insts_list] 
        other_cat_ids = [i.cat_ids for i in other_insts_list]
        other_bboxes = [i.bboxes for i in other_insts_list]
        other_masks = [i.masks for i in other_insts_list]

        new_confs = self.confs.concat(other_confs, update_flag)
        new_cat_ids = self.cat_ids.concat(other_cat_ids, update_flag)
        new_bboxes = self.bboxes.concat(other_bboxes, update_flag)

        if self.masks is not None:
            new_masks = self.masks.concat(other_masks, update_flag)
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