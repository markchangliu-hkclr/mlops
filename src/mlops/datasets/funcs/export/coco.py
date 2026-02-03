import os
import json
from pathlib import Path
import shutil
from typing import List, Dict, Literal

import pycocotools.mask as pycocomask

from mlops.datasets.typing.coco import CocoFileType, \
    CocoImgType, CocoAnnType, CocoCatType
from mlops.datasets.typing.image import ImgMetaType
from mlops.shapes.structs.instances import Instances
from mlops.shapes.funcs.convert.mask2poly import maskArr_to_polyCocos
from mlops.shapes.funcs.convert.poly2rle import polyCocos_to_rles


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

    img_dict: CocoImgType = {}
    ann_dict: CocoAnnType = {}
    cat_dict: CocoCatType = {}

    img_id = 0
    ann_id = 0
    
    for cat_id, cat_name in cat_id_name_dict.items():
        cat_dict["id"] = cat_id
        cat_dict["name"] = cat_name

    for meta, insts in zip(img_metas, insts_list):
        img_dict["file_name"] = Path(meta["img_p"]).name
        img_dict["height"] = meta["img_hw"][0]
        img_dict["width"] = meta["img_hw"][1]
        img_dict["id"] = img_id

        src_img_p = meta["img_p"]
        dst_img_p = os.path.join(export_img_dir, img_dict["file_name"])
        shutil.copy(src_img_p, dst_img_p)

        for inst in insts:
            bbox = inst.bboxes[0].tolist()
            ann_dict["bbox"] = bbox
            ann_dict["id"] = ann_id
            ann_dict["image_id"] = img_id
            ann_dict["iscrowd"] = False
            ann_dict["category_id"] = inst.cat_ids[0].tolist()

            if shape_format == "bbox":
                ann_dict["segmentation"] = []

                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                ann_dict["area"] = w * h
            else:
                mask = inst.masks[0]
                polys = maskArr_to_polyCocos(mask, True, False)
                ann_dict["segmentation"] = polys

                rles = polyCocos_to_rles(polys, meta["img_hw"], True)
                area = pycocomask.area(rles)
                ann_dict["area"] = area
            
            ann_id += 1
        
        img_id += 1
    
    

