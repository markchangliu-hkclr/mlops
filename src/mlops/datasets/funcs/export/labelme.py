import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Literal

from mlops.datasets.typing.image import ImgMetaType
from mlops.datasets.funcs.convert.insts2labelme import instances_to_labelmeFile
from mlops.shapes.structs.instances import Instances


__all__ = [
    "export_labelme"
]


def export_labelme(
    img_metas: List[ImgMetaType],
    insts_list: List[Instances],
    cat_id_name_dict: Dict[int, str],
    shape_format: Literal["bbox", "poly"],
    export_img_dir: str,
    export_label_dir: str
) -> None:
    os.makedirs(export_img_dir, exist_ok = True)
    os.makedirs(export_label_dir, exist_ok = True)

    data_id = 0

    for img_meta, insts in zip(img_metas, insts_list):
        img_p = img_meta["img_p"]
        img_suffix = Path(img_p).suffix
        dst_img_name = f"{data_id}{img_suffix}"
        dst_img_p = os.path.join(export_img_dir, dst_img_name)
        shutil.copy(img_p, dst_img_p)

        labelme_dict = instances_to_labelmeFile(
            dst_img_p, img_meta["img_hw"], insts,
            shape_format, cat_id_name_dict
        )

        with open(labelme_dict, "r") as f:
            json.dump(labelme_dict, f)
