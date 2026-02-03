import os
import json
from typing import List, Tuple, Dict, Literal

from mlops.datasets.funcs.convert.coco2insts import cocoAnns_to_instances
from mlops.datasets.typing.image import ImgMetaType
from mlops.datasets.typing.coco import CocoFileType, CocoAnnType
from mlops.shapes.structs.instances import Instances


__all__ = [
    "load_coco_dataset"
]


def load_coco_dataset(
    coco_fp: str,
    img_p_prefix: str,
    shape_format: Literal["bbox", "poly"],
) -> Tuple[List[ImgMetaType], List[Instances]]:
    with open(coco_fp, "r") as f:
        coco_dict: CocoFileType = json.load(f)

    img_id_meta_dict: Dict[int, ImgMetaType] = {}
    
    for img_dict in coco_dict["images"]:
        img_id = img_dict["id"]
        img_hw = (img_dict["height"], img_dict["width"])
        img_p = os.path.join(img_p_prefix, img_dict["file_name"])
        img_meta = {"img_hw": img_hw, "img_p": img_p}

        img_id_meta_dict[img_id] = img_meta
    
    img_id_anns_dict: Dict[int, List[CocoAnnType]] = {}

    for ann in coco_dict["annotations"]:
        img_id = ann["image_id"]

        if img_id in img_id_anns_dict.keys():
            img_id_anns_dict[img_id].append(ann)
        else:
            img_id_anns_dict[img_id] = [ann]
    
    img_metas = []
    insts_list = []
    for img_id, img_meta in img_id_meta_dict.items():
        img_metas.append(img_meta)

        anns = img_id_meta_dict[img_id]
        insts = cocoAnns_to_instances(anns, img_meta["img_hw"], shape_format)
        insts_list.append(insts)
    
    return img_metas, insts_list