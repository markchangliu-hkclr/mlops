import os
from typing import List, Literal, Dict, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from mlops.datasets.structs.metrics import PrecRecTable, MAPTable
from mlops.shapes.structs.instances import Instances
from mlops.shapes.funcs.eval.ious import get_ious
from mlops.shapes.funcs.eval.match import match


def get_tp_fn_single_img(
    gt_insts: Instances,
    pred_insts: Instances,
    iou_thres: float,
    shape_format: Literal["bbox", "poly"],
    match_mode: Literal["1vs1", "prec_focus", "rec_focus"],
) -> Tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """
    Single category evaluation.

    Return
    -----
    - `tp_pred_flags: NDArray[np.bool_], (num_preds, )`
    - `fn_gt_flags: NDArray[np.bool_], (num_gts, )`
    """
    assert shape_format in ["bbox", "poly"]
    assert match_mode in ["1vs1", "prec_focus", "rec_focus"]
    
    if shape_format == "bbox":
        pred_bboxes = pred_insts.bboxes
        gt_bboxes = gt_insts.bboxes
        ious = get_ious(gt_bboxes, pred_bboxes, "bbox", "iou")
    else:
        pred_masks = pred_insts.masks
        gt_masks = gt_insts.masks
        ious = get_ious(gt_masks, pred_masks, "poly", "iou")

    pred_matches, gt_matches = match(ious, iou_thres, match_mode)
    
    if len(pred_matches.shape) == 1:
        tp_pred_flags = pred_matches > -1
    elif len(pred_matches.shape) == 2:
        tp_pred_flags = np.any(pred_matches, axis = 1)
    else:
        raise ValueError("pred_matches")

    if len(gt_matches.shape) == 1:
        fn_gt_flags = gt_matches
    elif len(gt_matches.shape) == 2:
        fn_gt_flags = np.any(gt_matches, axis = 1)
    else:
        raise ValueError("gt_matches")

    return tp_pred_flags, fn_gt_flags

def get_PR_table(
    confs_list: List[NDArray[np.bool_]],
    tp_flags_list: List[NDArray[np.bool_]]
) -> PrecRecTable:
    """
    Single category evaluation.
    """
    confs = np.concat([c for c in confs_list])
    tp_pred_flags = np.concat([f for f in tp_flags_list])
    num_preds = len(tp_pred_flags)

    sort_indice = np.argsort(confs)[::-1]
    confs = confs[sort_indice]
    tp_pred_flags = tp_pred_flags[sort_indice]

    pred_counts = np.arange(num_preds) + 1
    cum_tps = np.cumsum(tp_pred_flags)
    precs = cum_tps / pred_counts
    recs = cum_tps / pred_counts

    # COCO-style interpolation
    recs_smooth = np.concatenate([[0], recs, [1]])
    precs_smooth = np.concatenate([[0], precs, [0]])
    precs_smooth = np.maximum.accumulate(precs[::-1])[::-1]

    prec_rec_table = PrecRecTable(
        confs, cum_tps, precs, precs_smooth, recs, recs_smooth
    )

    return prec_rec_table
    

def get_mAP(
    cat_ids: List[int],
    pr_tables: List[PrecRecTable]
) -> MAPTable:
    """
    Attrs
    -----
    - `cat_ids: List[int]`
    - `pr_tables: List[PrecRecTable], (num_cats, )`
    """
    assert len(cat_ids) == len(pr_tables)

    aps = {}
    map = 0
    for cat_id, pr_table in zip(cat_ids, pr_tables):
        ap = np.trapezoid(pr_table.precs_smooth, pr_table.recs_smooth)
        aps[cat_id] = ap
        map += ap
    
    map = map / len(cat_id)
    map_table = MAPTable(aps, map)
    return map_table

def get_conf_acc_table(
    pr_table: PrecRecTable
) -> 