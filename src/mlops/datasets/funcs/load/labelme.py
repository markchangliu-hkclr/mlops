import os
import json
from pathlib import Path
from typing import List, Tuple, Literal, Dict

from mlops.datasets.typing.labelme import LabelmeFileType
from mlops.shapes.structs.instances import Instances
from mlops.datasets.typing.image import ImgMetaType

from mlops.datasets.funcs.convert.labelme2insts import labelmeFile_to_instances


__all__ = [
    "load_labelme_dataset",
]


def load_labelme_dataset(
    img_dirs: List[str],
    labelme_dirs: List[str],
    cat_name_id_dict: Dict[str, int],
    merge_group_flag: bool,
    shape_format: Literal["bbox", "poly"],
) -> Tuple[List[ImgMetaType], List[Instances]]:
    """
    Returns
    -----
    - `img_metas: List[ImgMetaType]`
    - `insts_list: List[Instances]`
    """
    assert len(img_dirs) == len(labelme_dirs)

    img_metas = []
    insts_list = []

    for img_dir, lb_dir in zip(img_dirs, labelme_dirs):
        fns = os.listdir(lb_dir)

        for fn in fns:
            if fn.endswith(".json"):
                continue

            fp = os.path.join(lb_dir, fn)
            img_meta, insts = labelmeFile_to_instances(
                fp, merge_group_flag, cat_name_id_dict, shape_format
            )
            insts_list.append(insts)

            img_name = Path(fn).stem
            img_p = os.path.join(img_dir, img_name)

            with open(fp, "r") as f:
                labelme_dict: LabelmeFileType = json.load(f)
                img_h = labelme_dict["imageHeight"]
                img_w = labelme_dict["imageWidth"]
            
            img_meta: ImgMetaType = {
                "img_hw": (img_h, img_w),
                "img_p": img_p
            }
            img_metas.append(img_meta)
    
    return img_metas, insts_list