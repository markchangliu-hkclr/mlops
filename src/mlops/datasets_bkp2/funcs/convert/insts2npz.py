import os
from typing import Union, Literal

import numpy as np

from mlops.shapes.structs.instances import Instances


__all__ = [
    "instances_to_npz"
]


def instances_to_npz(
    insts: Instances,
    export_npz_p: Union[os.PathLike, str],
    shape_format: Literal["bbox", "poly"]
) -> None:
    assert shape_format in ["bbox", "poly"]

    bboxes = insts.bboxes
    scores = insts.confs

    if shape_format == "bbox":
        np.savez_compressed(
            export_npz_p, bboxes = bboxes, scores = scores
        )
    elif shape_format == "poly":
        masks = insts.masks.astype(np.bool_)
        np.savez_compressed(
            export_npz_p, bboxes = bboxes, scores = scores, masks = masks
        )
    