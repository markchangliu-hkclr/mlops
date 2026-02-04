import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Literal, Union

import cv2
import numpy as np
from numpy.typing import NDArray

from mlops.datasets.typing.image import ImgMetaType
from mlops.datasets.funcs.load.labelme import load_labelme
from mlops.datasets.funcs.load.coco import load_coco
from mlops.datasets.funcs.load.yolo import load_yolo
from mlops.datasets.funcs.export.labelme import export_labelme
from mlops.datasets.funcs.export.coco import export_coco
from mlops.datasets.funcs.export.yolo import export_yolo
from mlops.shapes.structs.instances import Instances
from mlops.shapes.funcs.visualize import draw_instances


__all__ = [
    "DetDataset"
]


@dataclass
class DetDataset:
    img_metas: List[ImgMetaType]
    insts_list: List[Instances]
    cat_name_id_dict: Dict[str, int]
    cat_id_name_dict: Dict[int, str]
    shape_format: Literal["bbox", "poly"]

    @classmethod
    def from_labelme(
        cls, 
        img_dirs: List[str],
        labelme_dirs: List[str],
        cat_name_id_dict: Dict[str, int],
        merge_group_flag: bool,
        shape_format: Literal["bbox", "poly"]
    ) -> "DetDataset":
        cat_id_name_dict = {v:k for k, v in cat_name_id_dict.items()}
        img_metas, insts_list = load_labelme(
            img_dirs, labelme_dirs, cat_name_id_dict, merge_group_flag,
            shape_format
        )

        dataset = DetDataset(
            img_metas, insts_list, cat_name_id_dict, cat_id_name_dict,
            shape_format
        )

        return dataset
    
    @classmethod
    def from_coco(
        cls,
        coco_fp: str,
        img_p_prefix: str,
        cat_name_id_dict: Dict[str, int],
        shape_format: Literal["bbox", "poly"]
    ) -> "DetDataset":
        img_metas, insts_list = load_coco(
            coco_fp, img_p_prefix, shape_format
        )

        cat_id_name_dict = {v:k for k, v in cat_name_id_dict.items()}
        
        dataset = DetDataset(
            img_metas, insts_list, cat_name_id_dict, 
            cat_id_name_dict, shape_format
        )

        return dataset
    
    @classmethod
    def from_yolo(
        cls,
        img_dirs: List[str],
        label_dirs: List[str],
        cat_name_id_dict: Dict[str, int],
        shape_format: Literal["bbox", "poly"]
    ) -> "DetDataset":
        img_metas, insts_list = load_yolo(
            img_dirs, label_dirs, shape_format
        )

        cat_id_name_dict = {v:k for k, v in cat_name_id_dict.items()}

        dataset = DetDataset(
            img_metas, insts_list, cat_name_id_dict, 
            cat_id_name_dict, shape_format
        )

        return dataset
    
    def __len__(self) -> int:
        return len(self.img_metas)
    
    def __getitem__(
        self,
        item: Union[int, List[int], NDArray[np.integer], NDArray[np.bool_], slice],
    ) -> "DetDataset":
        if isinstance(item, int):
            item = [item]
        elif isinstance(item, np.ndarray):
            item = np.arange(len(self))[item].tolist()
        
        assert isinstance(item, list)
        
        new_img_metas = []
        new_insts_list = []

        for i in item:
            new_img_metas.append(self.img_metas[i])
            new_insts_list.append(self.insts_list[i])
        
        new_dataset = DetDataset(
            new_img_metas, new_insts_list, self.cat_name_id_dict,
            self.cat_id_name_dict, self.shape_format
        )

        return new_dataset
    
    def concat(
        self, 
        other_datasets: List["DetDataset"],
        update_flag: bool
    ) -> "DetDataset":
        new_img_metas = [self.img_metas]
        new_insts_list = [self.insts_list]

        for dataset in other_datasets:
            assert len(self.cat_id_name_dict) == len(dataset.cat_id_name_dict)
            assert self.shape_format == dataset.shape_format

            new_img_metas.append(dataset.img_metas)
            new_insts_list.append(dataset.insts_list)
        
        if update_flag:
            self.img_metas = new_img_metas
            self.insts_list = new_insts_list
            return self
        else:
            new_dataset = DetDataset(
                new_img_metas, new_insts_list, self.cat_name_id_dict,
                self.cat_id_name_dict, self.shape_format
            )
            return new_dataset
    
    def draw(
        self,
        vis_img_dir: str,
    ) -> None:
        if os.path.exists(vis_img_dir):
            shutil.rmtree(vis_img_dir)

        os.makedirs(vis_img_dir, exist_ok = True)

        data_id = 0
        for img_meta, insts in zip(self.img_metas, self.insts_list):
            bgr = cv2.imread(img_meta["img_p"])
            img_vis = draw_instances(
                bgr, insts, "default", "default", "default"
            )

            img_suffix = Path(img_meta["img_p"]).suffix
            img_vis_name = f"{data_id}{img_suffix}"
            img_vis_p = os.path.join(vis_img_dir, img_vis_name)
            cv2.imwrite(img_vis_p, img_vis)
            
            data_id += 1
    
    def to_labelme(
        self,
        export_img_dir: str,
        export_label_dir: str
    ) -> None:
        export_labelme(
            self.img_metas, self.insts_list, self.cat_id_name_dict,
            self.shape_format, export_img_dir, export_label_dir
        )
    
    def to_coco(
        self,
        export_root: str,
        export_json_fn: str,
        export_img_dn: str
    ) -> None:
        export_coco(
            self.img_metas, self.insts_list, self.cat_id_name_dict,
            self.shape_format, export_root, export_json_fn, export_img_dn
        )
    
    def to_yolo(
        self,
        export_img_dir: str,
        export_label_dir: str
    ) -> None:
        export_yolo(
            self.img_metas, self.insts_list, self.shape_format,
            export_img_dir, export_label_dir
        )
    
    