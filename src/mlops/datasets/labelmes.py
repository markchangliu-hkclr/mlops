import os
from typing import List, Literal

from mlops.datasets.types import LabelmeFileType, LabelmeShapeType
from mlops.datasets.datasets import 


def convert_labelme2coco(
    labelme_dirs: List[str],
    img_dirs: List[str],
    merge_group_flag: List[str],
    shape_type: Literal["bbox", "poly"],
    
)