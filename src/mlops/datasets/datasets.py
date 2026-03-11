import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Dict, Literal

import numpy as np

from mlops.datasets.types import CocoImgType, CocoAnnType, CocoCatType, CocoFileType
from mlops.datasets.types import LabelmeFileType, LabelmeShapeType
from mlops.datasets.types import YoloDetAnnType, YoloSegAnnType
from mlops.datasets.bboxes import convert_bboxLabelme2bboxCoco
from mlops.datasets.polygons import convert_polyLabelme2polyArr
from mlops.datasets.polygons import convert


@dataclass
class DetDataset:
    img_prefix: Union[str, None]
    imgs_list: List[CocoImgType]
    anns_list: List[List[CocoAnnType]]
    cat_name_id_dict: Dict[str, int]
    cat_id_name_dict: Dict[int, str]
    shape_format: Literal["bbox", "poly"]

    def __post_init__(self) -> None:
        assert len(self.imgs_list) == len(self.anns_list)
    
    def __len__(self) -> int:
        return len(self.imgs_list)
    
    def get_subset(
        self,
        cat_ids: List[int],
    ) -> "DetDataset":
        new_imgs_list = []
        new_anns_list = []
        target_cat_ids = cat_ids

        for i in range(len(self.imgs_list)):
            img = self.imgs_list[i]
            anns = self.anns_list[i]
            new_anns = []

            for ann in anns:
                cat_id = ann["category_id"]
                if cat_id in target_cat_ids:
                    new_anns.append(ann)
            
            if len(new_anns) > 0:
                new_imgs_list.append(img)
                new_anns_list.append(new_anns)
        
        new_dataset = DetDataset(
            self.img_prefix, new_imgs_list, new_anns_list,
            self.cat_name_id_dict, self.cat_id_name_dict, self.shape_format
        )

        return new_dataset

def convert_labelmeFile2cocoAnn(
    label_p: str,
    merge_group_flag: bool,
    cat_name_id_dict: Dict[str, int],
    shape_format: Literal["bbox", "poly"],
    img_id: int,
    start_ann_id: int
) -> List[CocoAnnType]:
    with open(label_p, "r") as f:
        labelme_dict: LabelmeFileType = json.load(f)
    
    anns: List[CocoAnnType] = []
    shapes: List[LabelmeShapeType] = labelme_dict["shapes"]

    for shape in shapes:
        cat_id = cat_name_id_dict[shape["label"]]

        if shape_format == "bbox":
            bbox = shape["points"]
            bbox = convert_bboxLabelme2bboxCoco(bbox)
            poly = []
        else:
            poly = shape["points"]
            if len(poly) == 2:
                poly.append(poly[-1])
            poly = convert_polyLabelme2polyArr(poly)
            x1 = np.min(poly[:, 0]).item()
            x2 = np.max(poly[:, 0]).item()
            y1 = np.min(poly[:, 1]).item()
            y2 = np.max(poly[:, 1]).item()
            w = x2 - x1
            h = y2 - y1
            bbox = [x1, y1, w, h]
            poly = convert(poly)


def load_labelme(
    labelme_roots: List[str],
    img_roots: List[str],
    merge_group_flag: bool,
    cat_name_id_dict: Dict[str, int],
    shape_format: Literal["bbox", "poly"],
    img_suffixes: List[str],
    img_prefix: Union[str, None]
) -> "DetDataset":
    assert isinstance(labelme_roots, list)
    assert isinstance(img_roots, list)
    assert len(labelme_roots) == len(img_roots)

    img_ps = []
    label_ps = []
    for l_root, i_root in zip(labelme_roots, img_roots):
        for root, subdirs, files in os.walk(i_root):
            for file in files:
                if not file.endswith(tuple(img_suffixes)):
                    continue
                
                stem = Path(file).stem
                img_p = os.path.join(root, file)
                rel_dir = str(Path(img_p).relative_to(i_root).parent)

                label_name = f"{stem}.json"
                label_p = os.path.join(l_root, rel_dir, label_name)

                if not os.path.exists(label_p):
                    continue

                img_ps.append(img_p)
                label_ps.append(label_p)

    for img_p, label_p in zip(img_ps, label_ps):
        coco_img: CocoImgType = {}
        coco_anns: List[CocoAnnType] = []

        with open(label_p, "r") as f:
            labelme_dict: LabelmeFileType = json.load(f)
        
        if img_prefix is not None:
            coco_img["file_name"] = str(Path(img_p).relative_to(img_prefix))
        else:
            coco_img["file_name"] = img_p

        coco_img["height"] = labelme_dict["imageHeight"]
        coco_img["width"] = 