from typing import List, Literal, Tuple

import pycocotools.mask as pycocomask

from mlops.datasets.typing.coco import CocoAnnType
from mlops.shapes.structs.instances import Instances
from mlops.shapes.funcs.convert.mask2poly import maskArr_to_polyCocos
from mlops.shapes.funcs.convert.poly2rle import polyCocos_to_rles


__all__ = [
    "instances_to_cocoAnns"
]


def instances_to_cocoAnns(
    insts: Instances,
    img_hw: Tuple[int, int],
    shape_format: Literal["bbox", "poly"],
    img_id: int,
    start_ann_id: int
) -> List[CocoAnnType]:
    ann_dicts = []
    ann_id = start_ann_id

    for inst in insts:
        ann_dict: CocoAnnType = {}

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

            rles = polyCocos_to_rles(polys, img_hw, True)
            area = pycocomask.area(rles)
            ann_dict["area"] = area
        
        ann_dicts.append(ann_dict)
        ann_id += 1
    
    return ann_dicts

