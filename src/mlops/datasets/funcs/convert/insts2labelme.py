from pathlib import Path
from typing import Literal, Dict, Tuple

from mlops.datasets.typing.labelme import LabelmeFileType, LabelmeShapeType
from mlops.shapes.structs.instances import Instances
from mlops.shapes.funcs.convert.bbox2bbox import bboxesArrXYXY_to_bboxesLabelme
from mlops.shapes.funcs.convert.mask2poly import maskArr_to_polyLabelmes


__all__ = [
    "instances_to_labelmeFile"
]


def instances_to_labelmeFile(
    dst_img_p: str,
    img_hw: Tuple[int, int],
    insts: Instances,
    shape_format: Literal["bbox", "poly"],
    cat_id_name_dict: Dict[int, str],
) -> LabelmeFileType:
    assert shape_format in ["bbox", "poly"]

    labelme_dict: LabelmeFileType = {
        "version": "5.4.1",
        "flags": {},
        "shapes": [],
        "imageData": None,
        "imageHeight": img_hw[0],
        "imageWidth": img_hw[1],
        "imagePath": Path(dst_img_p).name
    }

    if shape_format == "bbox":
        bboxes = insts.bboxes
        bboxes = bboxesArrXYXY_to_bboxesLabelme(bboxes)
    else:
        masks = insts.masks
        polys_list = []

        for mask in masks:
            polys = maskArr_to_polyLabelmes(mask, True, False)
            polys_list.append(polys)
    
    for i in range(len(insts)):
        cat_id = insts.cat_ids[i].item()
        cat_name = cat_id_name_dict[cat_id]

        shape_dict: LabelmeShapeType = {
            "flags": {},
            "label": cat_name
        }

        if shape_format == "bbox":
            bbox = bboxes[i]
            shape_dict["group_id"] = None,
            shape_dict["points"] = bbox
            shape_dict["shape_type"] = "rectangle"

            labelme_dict["shapes"].append(shape_dict)
        else:
            polys = polys_list[i]
            
            for poly in polys:
                shape_dict["group_id"] = i
                shape_dict["points"] = poly
                shape_dict["shape_type"] = "polygon"

                labelme_dict["shapes"].append(shape_dict)
    
    return labelme_dict
                