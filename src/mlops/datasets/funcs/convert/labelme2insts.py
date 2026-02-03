import json
from typing import List, Tuple, Literal, Dict

import numpy as np

from mlops.datasets.typing.labelme import LabelmeFileType, \
    LabelmeShapeGroupsType, LabelmeShapeType
from mlops.shapes.structs.instances import Instances
from mlops.shapes.funcs.convert.poly2mask import polyLabelme_to_maskArr
from mlops.shapes.funcs.convert.bbox2bbox import bboxLabelme_to_bboxArrXYXY
from mlops.shapes.funcs.convert.poly2bbox import polyLabelme_to_bboxLabelme
from mlops.shapes.funcs.merge.others import merge_masksArr, merge_bboxesArrXYXY
from mlops.shapes.funcs.concat import concat_instances


__all__ = [
    "labelmeFile_to_instances",
]


def get_shape_groups(
    labelme_dict: LabelmeFileType
) -> LabelmeShapeGroupsType:
    shape_groups = {}
    shape_list: List[LabelmeShapeType] = labelme_dict["shapes"]

    for i, shape in enumerate(shape_list):
        group_id = shape["group_id"]

        if group_id is None:
            shape_groups[f"shape{i}"] = [shape]
        elif group_id in shape_groups.keys():
            shape_groups[group_id].append(shape)
        else:
            shape_groups[group_id] = [shape]
    
    return shape_groups

def shapeGroup_to_instances(
    shape_group: List[LabelmeShapeType],
    img_hw: Tuple[int, int],
    merge_group_flag: bool,
    cat_name_id_dict: Dict[str, int],
    shape_format: Literal["bbox", "poly"],
) -> Instances:
    if len(shape_group) == 0:
        bboxes = np.zeros(0).astype(np.int32)
        masks = np.zeros((0, img_hw[0], img_hw[1])).astype(np.bool_)
        cat_ids = np.zeros(0).astype(np.int32)
        confs = np.ones(0).astype(np.float32)
        insts = Instances(confs, cat_ids, bboxes, masks)
        return insts
    
    bboxes = []
    masks = []
    cat_ids = []

    for shape in shape_group:
        cat_name = shape["label"]
        cat_id = cat_name_id_dict[cat_name]
        cat_ids.append(cat_id)

        if shape_format == "bbox":
            bbox = shape["points"]
            bbox = bboxLabelme_to_bboxArrXYXY(bbox)
            bboxes.append(bbox)
        else:
            poly = shape["points"]
            bbox = polyLabelme_to_bboxLabelme(poly)
            bbox = bboxLabelme_to_bboxArrXYXY(bbox)
            bboxes.append(bbox)
            
            mask = polyLabelme_to_maskArr(poly, img_hw)
            masks.append(mask)
    
    confs = np.ones(len(bboxes)).astype(np.float32)
    cat_ids = np.asarray(cat_ids).astype(np.int32)
    bboxes = np.stack(bboxes, axis = 0).astype(np.int32)

    if shape_format == "poly":
        masks = np.stack(masks, axis = 0).astype(np.bool_)
        
    if merge_group_flag:
        bboxes = merge_bboxesArrXYXY(bboxes)[None, ...]
        masks = merge_masksArr(masks)[None, ...]
        cat_ids = cat_ids[[0]]
        confs = confs[[0]]

    insts = Instances(confs, cat_ids, bboxes, masks)
    return insts

def labelmeFile_to_instances(
    fp: str,
    merge_group_flag: bool,
    cat_name_id_dict: Dict[str, int],
    shape_format: Literal["bbox", "poly"],
) -> Instances:
    with open(fp, "r") as f:
        labelme_dict: LabelmeFileType = json.load(f)

    shape_groups = get_shape_groups(labelme_dict)
    img_h = labelme_dict["imageHeight"]
    img_w = labelme_dict["imageWidth"]
    insts_list = []

    for shape_group in shape_groups:
        insts = shapeGroup_to_instances(
            shape_group, (img_h, img_w), merge_group_flag,
            cat_name_id_dict, shape_format
        )
        insts_list.append(insts)
    
    insts = concat_instances(insts_list)
    return insts

