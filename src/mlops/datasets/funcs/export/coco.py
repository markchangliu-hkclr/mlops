import os
import json
from pathlib import Path
import shutil
from typing import List, Dict, Literal

from mlops.datasets.typing.coco import CocoFileType, CocoImgType, CocoCatType
from mlops.datasets.typing.image import ImgMetaType
from mlops.shapes.structs.instances import Instances

from mlops.datasets.funcs.convert.insts2coco import instances_to_cocoAnns


__all__ = [
    "export_coco"
]


def export_coco(
    img_metas: List[ImgMetaType],
    insts_list: List[Instances],
    cat_id_name_dict: Dict[int, str],
    shape_format: Literal["bbox", "poly"],
    export_root: str,
    export_json_fn: str,
    export_img_dn: str
) -> None:
    assert shape_format in ["bbox", "poly"]

    export_img_dir = os.path.join(export_root, export_img_dn)
    export_json_p = os.path.join(export_root, export_json_fn)
    os.makedirs(export_img_dir, exist_ok = True)

    coco_dict: CocoFileType = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    img_id = 0
    ann_id = 0
    
    for cat_id, cat_name in cat_id_name_dict.items():
        cat_dict: CocoCatType = {}
        cat_dict["id"] = cat_id
        cat_dict["name"] = cat_name
        coco_dict["categories"].append(cat_dict)

    for meta, insts in zip(img_metas, insts_list):
        img_dict: CocoImgType = {}
        img_dict["file_name"] = Path(meta["img_p"]).name
        img_dict["height"] = meta["img_hw"][0]
        img_dict["width"] = meta["img_hw"][1]
        img_dict["id"] = img_id
        coco_dict["images"].append(img_dict)

        src_img_p = meta["img_p"]
        dst_img_p = os.path.join(export_img_dir, img_dict["file_name"])
        shutil.copy(src_img_p, dst_img_p)

        anns = instances_to_cocoAnns(
            insts, meta["img_hw"], shape_format, img_id, ann_id
        )
        coco_dict["annotations"] += anns
        
        img_id += 1
        ann_id += len(anns)
    
    with open(export_json_p, "w") as f:
        json.dump(coco_dict, f)
    
    

