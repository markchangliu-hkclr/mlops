import os
from typing import List, Literal, Dict, Tuple

from mlops.datasets.typing.image import ImgMetaType
from mlops.shapes.structs.instances import Instances
from mlops.shapes.funcs.eval.ious import iou_masks, iou_bboxes
from mlops.shapes.funcs.eval.match import match_1vs1, \
    match_prec_focus, match_rec_focus, get_tp_fp_fn





def eval_dataset(
    img_metas: List[ImgMetaType],
    gt_insts_list: List[Instances],
    pred_insts_list: List[Instances],
    iou_thres: float,
    shape_format: Literal["bbox", "poly"],
    export_res_p: str,
    match_mode: Literal["1vs1", "prec_focus", "rec_focus"]
) -> Tuple[Dict[str, float], Tuple[Dict[str, float]], Tuple[Dict[str, float]]]:
    """
    Returns
    -----
    - `precs: Dict[str, float], {cat_name: prec}`
    - `recs: Dict[str, float], {cat_name: rec}`
    - `aps: Dict[str, float], {cat_name: rec}`
    """
    assert shape_format in ["bbox", "poly"]
    assert len(img_metas) == len(gt_insts_list) == len(pred_insts_list)

    for i in range(len(img_metas)):
        img_meta = img_metas[i]
        gt_insts = gt_insts_list[i]
        pred_insts = pred_insts_list[i]

        if shape_format == "bbox":
            gt_bboxes = gt_insts.bboxes
            pred_bboxes = pred_insts.bboxes