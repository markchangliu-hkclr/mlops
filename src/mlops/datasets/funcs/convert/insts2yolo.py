import os
from typing import List, Union, Literal

from mlops.datasets.typing.image import ImgMetaType
from mlops.datasets.typing.yolo import YoloBBoxType, YoloPolyType
from mlops.shapes.structs.instances import Instances


def instances_to_yoloFile(
    img_meta: ImgMetaType,
    insts: Instances,
    shape_format: Literal["bbox", "poly"]
) -> Union[List[YoloBBoxType], List[YoloPolyType]]