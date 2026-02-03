from typing import List, Union, Literal, Tuple

from mlops.datasets.typing.yolo import YoloBBoxType, YoloPolyType
from mlops.shapes.structs.instances import Instances
from mlops.shapes.funcs.convert.bbox2bbox import bboxesArrXYXY_to_bboxesYolo
from mlops.shapes.funcs.convert.mask2poly import maskArr_to_polyYolos


__all__ = [
    "instances_to_yoloFile"
]


def instances_to_yoloFile(
    img_hw: Tuple[int, int],
    insts: Instances,
    shape_format: Literal["bbox", "poly"]
) -> Union[List[YoloBBoxType], List[YoloPolyType]]:
    assert shape_format in ["bbox", "poly"]

    if len(insts) == 0:
        return []

    if shape_format == "bbox":
        bboxes = insts.bboxes
        bboxes = bboxesArrXYXY_to_bboxesYolo(bboxes, img_hw)
        return bboxes
    else:
        masks = insts.masks
        polys = []

        for mask in masks:
            poly = maskArr_to_polyYolos(mask, True, True, img_hw)[0]
            polys.append(poly)
        return polys