import os
import shutil
from pathlib import Path
from typing import List, Literal

from mlops.datasets.funcs.convert.insts2npz import instances_to_npz
from mlops.datasets.typing.image import ImgMetaType
from mlops.shapes.structs.instances import Instances


def export_npz(
    img_metas: List[ImgMetaType],
    insts_list: List[Instances],
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

        dst_npz_p = os.path.join(export_label_dir, f"{data_id}.npz")
        instances_to_npz(insts, dst_npz_p, shape_format)

        data_id += 1