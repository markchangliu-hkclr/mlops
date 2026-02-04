import os
import shutil
from typing import List, Literal
from pathlib import Path

from mlops.datasets.funcs.convert.insts2yolo import instances_to_yoloFile
from mlops.datasets.typing.image import ImgMetaType
from mlops.shapes.structs.instances import Instances


__all__ = [
    "export_yolo"
]


def export_yolo(
    img_metas: List[ImgMetaType],
    insts_list: List[Instances],
    shape_format: Literal["bbox", "poly"],
    export_img_dir: str,
    export_label_dir: str
) -> None:
    assert shape_format in ["bbox", "poly"]

    os.makedirs(export_img_dir, exist_ok = True)
    os.makedirs(export_label_dir, exist_ok = True)

    data_id = 0

    for img_meta, insts in zip(img_metas, insts_list):
        lines = instances_to_yoloFile(
            img_meta["img_hw"], insts, shape_format
        )

        src_img_p = img_meta["img_p"]
        img_suffix = Path(src_img_p).suffix
        dst_img_name = f"{data_id}{img_suffix}"
        dst_img_p = os.path.join(export_img_dir, dst_img_name)
        shutil.copy(src_img_p, dst_img_p)
        
        dst_label_p = os.path.join(export_label_dir, f"{data_id}.txt")
        with open(dst_label_p, "w") as f:
            for line in lines:
                line = [str(e) for e in line]
                line = " ".join(line)
                f.write(f"{line}\n")
    
