from typing import Tuple, Literal

import numpy as np

from mlops.shapes.structs.instances import Instances
from mlops.shapes.funcs.convert.bbox2bbox import bboxYolo_to_bboxArrXYXY
from mlops.shapes.funcs.convert.poly2mask import polysYolo_to_maskArr
from mlops.shapes.funcs.convert.poly2bbox import polyYolo_to_bboxYolo


__all__ = [
    "yoloFile_to_instances"
]


def yoloFile_to_instances(
    fp: str,
    img_hw: Tuple[int, int],
    shape_format: Literal["bbox", "mask"]
) -> Instances:
    cat_ids = []
    bboxes = []
    masks = []

    with open(fp, "r") as f:
        for line in f:
            line = line.strip()
            line = line.split(" ")
            cat_id = int(line[0])
            cat_ids.append(cat_id)

            if shape_format == "bbox":
                x_ctr_norm = float(line[1])
                y_ctr_norm = float(line[2])
                w_norm = float(line[3])
                h_norm = float(line[4])
                bbox = [x_ctr_norm, y_ctr_norm, w_norm, h_norm]
                bbox = bboxYolo_to_bboxArrXYXY(bbox, img_hw)
                bboxes.append(bbox)
            else:
                poly = line[1:]
                poly = [float(i) for i in poly]

                mask = polysYolo_to_maskArr([poly], img_hw, False)[0]
                masks.append(mask)

                bbox = polyYolo_to_bboxYolo(poly)
                bbox = bboxYolo_to_bboxArrXYXY(bbox, img_hw)
                bboxes.append(bbox)
            
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