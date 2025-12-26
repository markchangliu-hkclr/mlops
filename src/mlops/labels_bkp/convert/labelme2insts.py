import json
import os
from typing import Dict, Union

import numpy as np
import pycocotools.mask as pycocomask

from mlops.shapes.insts import Insts


def labelme2insts_mask(
    labelme_p: Union[os.PathLike, str],
    cat_name_id_dict: Dict[str, int],
) -> Insts:
    with open(labelme_p, "r") as f:
        labelme_dict = json.load(f)
    
    anns = labelme_dict["shapes"]
    cat_ids = []
    bboxes = []
    masks = []

    img_h = labelme_dict["imageHeight"]
    img_w = labelme_dict["imageWidth"]

    for ann in anns:
        cat_id = cat_name_id_dict[ann["label"]]
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
    
    cat_ids = np.asarray(cat_ids, dtype=np.int32)
    scores = np.ones_like(cat_ids, dtype=np.float32)
    bboxes = np.asarray(bboxes, dtype=np.int32)

    if len(cat_ids) > 0:
        masks = np.concatenate(masks, axis=0).astype(np.bool_)
    else:
        masks = np.zeros((0, img_h, img_w)).astype(np.bool_)

    insts = Insts(scores, cat_ids, bboxes, masks)
    
    return insts

def labelme2insts_bbox(
    labelme_p: Union[os.PathLike, str],
    cat_name_id_dict: Dict[str, int],
) -> Insts:
    with open(labelme_p, "r") as f:
        labelme_dict = json.load(f)
    
    anns = labelme_dict["shapes"]
    cat_ids = []
    bboxes = []

    img_h = labelme_dict["imageHeight"]
    img_w = labelme_dict["imageWidth"]

    for ann in anns:
        cat_id = cat_name_id_dict[ann["label"]]
        x1y1, x2y2 = ann["points"]
        x1, y1 = x1y1
        x2, y2 = x2y2
        bbox = [x1, y1, x2, y2]

        bboxes.append(bbox)
        cat_ids.append(cat_id)
    
    cat_ids = np.asarray(cat_ids, dtype=np.int32)
    scores = np.ones_like(cat_ids, dtype=np.float32)
    bboxes = np.asarray(bboxes, dtype=np.int32)

    insts = Insts(scores, cat_ids, bboxes, None)
    
    return insts