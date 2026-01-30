import json
import os
from typing import Dict, Literal, Union

import numpy as np
import pycocotools.mask as pycocomask

from mlops.shapes.typing import BBoxLabelmeType
from mlops.shapes.objects import Instances
from mlops.datasets.typing.labelme import LabelmeDictType, LabelmeShapeType


def load_labelme(
    labelme_p: str,
    cat_name_id_dict: Dict[str, int],
    label_format: Literal["bbox", "mask"]
) -> Instances:
    if label_format == "mask":
        raise NotImplementedError

    
    