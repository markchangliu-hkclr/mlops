import os
from pathlib import Path
from typing import List, Literal, Tuple

import cv2

from mlops.datasets.funcs.convert.yolo2insts import yoloFile_to_instances
from mlops.datasets.typing.image import ImgMetaType
from mlops.shapes.structs.instances import Instances


__all__ = [
    "load_yolo"
]


def load_yolo(
    img_dirs: List[str],
    label_dirs: List[str],
    shape_format: Literal["bbox", "poly"],
) -> Tuple[List[ImgMetaType], List[Instances]]:
    assert isinstance(img_dirs, list)
    assert isinstance(label_dirs, list)
    assert len(img_dirs) == len(label_dirs)
    assert shape_format in Literal["bbox", "poly"]

    img_metas = []
    insts_list = []

    for img_dir, label_dir in zip(img_dirs, label_dirs):
        fns = os.listdir(img_dir)
        
        for fn in fns:
            if not fn.endswith((".png", ".jpg", ".jpeg")):
                continue

            img_p = os.path.join(img_dir, fn)
            img_stem = Path(img_p).stem
            label_p = os.path.join(label_dir, f"{img_stem}.txt")

            if not os.path.exists(label_p):
                continue
            
            img = cv2.imread(img_p)
            img_hw = img.shape[:2]
            insts = yoloFile_to_instances(label_p, img_hw, shape_format)

            img_meta: ImgMetaType = {
                "img_hw": img_hw,
                "img_p": img_p
            }

            img_metas.append(img_meta)
            insts_list.append(insts)
    
    return img_metas, insts_list
