import json
import os
from typing import Union, Dict, Literal, Tuple

import numpy as np

from mlops.labels.typedef.labelme import LabelmeDictType, LabelmeShapeDictType
from mlops.shapes.insts import Insts
from mlops.shapes.convert.bbox2bbox import bboxes2bboxes_xyxy2labelme
from mlops.shapes.convert.mask2poly import mask2poly_labelme


def insts2labelme(
    insts: Insts,
    img_name: str,
    img_hw: Tuple[int, int],
    export_labelme_p: Union[str, os.PathLike],
    cat_id_name_dict: Dict[int, str],
    shape_type: Literal["bbox", "mask"]
) -> None:
    assert shape_type in ["bbox", "mask"]

    labelme_dict: LabelmeDictType = {}
    labelme_dict["flags"] = {}
    labelme_dict["imagePath"] = img_name
    labelme_dict["imageData"] = None
    labelme_dict["imageHeight"] = img_hw[0]
    labelme_dict["imageWidth"] = img_hw[1]
    labelme_dict["version"] = "5.1.1"
    labelme_dict["shapes"] = []

    map_func = np.vectorize(
        lambda x: cat_id_name_dict.get(x)
    )
    cat_names = map_func(insts.cat_ids)

    if shape_type == "bbox":
        bboxes_labelme = bboxes2bboxes_xyxy2labelme(insts.bboxes)
    else:
        polys_labelme = []
        for mask in insts.masks:
            poly_labelme = mask2poly_labelme(mask, True)
            polys_labelme.append(poly_labelme)


    for i in range(len(insts)):
        shape: LabelmeShapeDictType = {}
        shape["flags"] = {}
        shape["group_id"] = None
        shape["label"] = cat_names[i].item()

        if shape_type == "bbox":
            shape["shape_type"] = "rectangle"
            shape["points"] = bboxes_labelme[i]
        else:
            shape["shape_type"] = "polygon"
            shape["points"] = polys_labelme[i]

        labelme_dict["shapes"].append(shape)

    with open(export_labelme_p, "w") as f:
        json.dump(labelme_dict, f)

