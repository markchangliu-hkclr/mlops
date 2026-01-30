import os
import json
from typing import List, Tuple, Literal, Dict

import numpy as np

from mlops.datasets.typing.labelme import LabelmeFileType, \
    LabelmeShapeGroupsType, LabelmeShapeType
from mlops.shapes.structs.instances import Instances
from mlops.shapes.funcs.convert.poly2rle import polyLabelmes_to_rles
from mlops.shapes.funcs.convert.rle2mask import rle_to_mask
from mlops.shapes.funcs.convert.bbox2bbox import bboxesCoco_to_bboxesArrXYXY
from mlops.shapes.funcs.convert.poly2bbox import polyLabelme_to_bboxLabelme
from mlops.shapes.funcs.merge.others import merge_rles


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
    merge_group_flag: bool,
    cat_name_id_dict: Dict[str, int],
    shape_format: Literal["bbox", "poly"],
) -> Instances:
    bboxes = []
    polys = []

    for shape in shape_group:
        if shape_format == "bbox":
            bbox = shape["points"]
            bboxes.append(bbox)
        else:
            poly = shape["points"]
            bbox.append

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

    for shape_group in shape_groups:


    for ann in anns:
        cat_id = cat_name_id_dict[ann["label"]]

        if shape_format == "poly":

            poly = np.asarray(ann["points"], dtype=np.int32) # (num_points, 2)
            poly = poly.flatten().tolist() # (num_points * 2, )
            rle = pycocomask.frPyObjects([poly], img_h, img_w)
            mask = pycocomask.decode(rle) # (img_h, img_w, 1)
            mask = np.transpose(mask, (2, 0, 1)) # (1, img_h, img_w)
            x1, y1, w, h = pycocomask.toBbox(rle).flatten().tolist()
            x2, y2 = x1 + w, y1 + h
            bbox = [x1, y1, x2, y2]

            masks.append(mask)
            cat_ids.append(cat_id)
            bboxes.append(bbox)


def load_labelme_dataset(
    img_dirs: List[str],
    labelme_dirs: List[str],
    cat_name_id_dict: Dict[str, int],
    shape_format: Literal["bbox", "poly"],
) -> Tuple[List[str], List[Instances]]:
    """
    Returns
    -----
    - `img_ps: List[str]`
    - `insts_list: List[Instances]`
    """
    pass