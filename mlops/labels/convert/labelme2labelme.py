import copy
import json
import os
from typing import Union, List

from mlops.labels.typedef.labelme import LabelmeDictType, LabelmeShapeDictType
from mlops.shapes.typedef.bboxes import BBoxesLabelmeType, BBoxLabelmeType
from mlops.shapes.typedef.polys import PolysLabelmeType
from mlops.shapes.convert.poly2bbox import polys2bbox_labelme
import mlops.labels.utils.labelme as labelme_utils


def labelme2labelme_poly2bbox_file(
    labelme_p: Union[str, os.PathLike],
    export_label_p: Union[str, os.PathLike],
) -> None:
    with open(labelme_p, "r") as f:
        labelme_dict: LabelmeDictType = json.load(f)

    export_labelme_dict = copy.deepcopy(labelme_dict)
    export_labelme_dict["shapes"] = []
    
    shape_groups = labelme_utils.get_shape_groups(
        labelme_dict
    )

    for group_id, group_shapes in shape_groups.items():
        group_polys = [s["points"] for s in group_shapes]
        bbox: BBoxLabelmeType = polys2bbox_labelme(group_polys)
        bbox_shape: LabelmeShapeDictType = {
            "points": bbox,
            "flags": group_shapes[0]["flags"],
            "group_id": None,
            "label": group_shapes[0]["label"],
            "shape_type": "rectangle"
        }

        export_labelme_dict["shapes"].append(bbox_shape)
    
    with open(export_label_p, "w") as f:
        json.dump(export_labelme_dict, f)

def labelme2labelme_poly2bbox_batch(
    labelme_dirs: List[str],
    export_labelme_dirs: List[str],
) -> None:
    assert isinstance(labelme_dirs, list)
    assert isinstance(export_labelme_dirs, list)
    assert len(labelme_dirs) == len(export_labelme_dirs)

    for l_dir, export_l_dir in zip(labelme_dirs, export_labelme_dirs):
        fns = os.listdir(l_dir)

        for fn in fns:
            if not fn.endswith(".json"):
                continue

            l_p = os.path.join(l_dir, fn)
            export_l_p = os.path.join(export_l_dir, fn)

            labelme2labelme_poly2bbox_file(
                l_p, export_l_p
            )