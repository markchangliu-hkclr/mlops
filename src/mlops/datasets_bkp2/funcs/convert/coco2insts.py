from typing import List, Literal, Tuple

import numpy as np

from mlops.datasets.typing.coco import CocoAnnType
from mlops.shapes.structs.instances import Instances
from mlops.shapes.funcs.convert.bbox2bbox import bboxCoco_to_bboxArrXYXY
from mlops.shapes.funcs.convert.poly2mask import polysCoco_to_masksArr


__all__ = [
    "cocoAnns_to_instances"
]


def cocoAnns_to_instances(
    coco_anns: List[CocoAnnType],
    img_hw: Tuple[int, int],
    shape_format: Literal["bbox", "poly"],
) -> Instances:
    assert shape_format in ["bbox", "poly"]
    
    bboxes = []
    masks = []
    cat_ids = []

    for ann in coco_anns:
        cat_id = ann["category_id"]
        cat_ids.append(cat_id)

        bbox = ann["bbox"]
        bbox = bboxCoco_to_bboxArrXYXY(bbox)
        bboxes.append(bbox)

        if shape_format == "poly":
            polys = ann["segmentation"]
            mask = polysCoco_to_masksArr(polys, img_hw, True)[0]
            masks.append(mask)
    
    if len(cat_ids) == 0:
        confs = np.zeros(0).astype(np.float32)
        cat_ids = np.zeros(0).astype(np.int32)
        bboxes = np.zeros((0, 4)).astype(np.int32)
    else:
        cat_ids = np.asarray(cat_ids).astype(np.int32)
        confs = np.ones(len(cat_ids)).astype(np.float32)
        bboxes = np.stack(bboxes, axis = 0).astype(np.int32)

    if shape_format == "bbox":
        masks = None
    elif len(cat_ids) == 0:
        masks = np.zeros((0, img_hw[0], img_hw[1])).astype(np.bool_)
    else:
        masks = np.stack(masks, axis = 0).astype(np.bool_)
    
    insts = Instances(confs, cat_ids, bboxes, masks)

    return insts