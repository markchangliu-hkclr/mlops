import os
from typing import List, Literal, Dict, Tuple, overload

import numpy as np
from numpy.typing import NDArray

from mlops.shapes.structs.instances import Instances
from mlops.shapes.funcs.eval.ious import iou_masks, iou_bboxes
from mlops.shapes.funcs.eval.match import match_1vs1, \
    match_prec_focus, match_rec_focus, get_tp_fn


def get_tp_fn_single_img(
    gt_insts_single_cat: Instances,
    pred_insts_single_cat: Instances,
    iou_thres: float,
    shape_format: Literal["bbox", "poly"],
    match_mode: Literal["1vs1", "prec_focus", "rec_focus"],
) -> Tuple[NDArray[np.floating], NDArray[np.bool_], NDArray[np.bool_]]:
    assert shape_format in ["bbox", "poly"]
    assert match_mode in ["1vs1", "prec_focus", "rec_focus"]
    
    if shape_format == "bbox":
        pred_bboxes = pred_insts_single_cat.bboxes
        gt_bboxes = gt_insts_single_cat.bboxes
        ious = iou_bboxes(gt_bboxes, pred_bboxes)
    else:
        pred_masks = pred_insts_single_cat.masks
        gt_masks = gt_insts_single_cat.masks
        ious = iou_masks(gt_masks, pred_masks)
    
    if match_mode == "1vs1":
        pred_matches, gt_matches = match_1vs1(ious, iou_thres)
    elif match_mode == "prec_focus":
        pred_matches, gt_matches = match_prec_focus(ious, iou_thres)
    elif match_mode == "rec_focus":
        pred_matches, gt_matches = match_rec_focus(ious, iou_thres)
    
    tp_pred_flags, fn_gt_flags = get_tp_fn(pred_matches, gt_matches)

    result = DetResult(pred_insts_single_cat.confs, tp_pred_flags, fn_gt_flags)

    return result

def get_conf_prec_rec_table(
    confs: List[]
    results: List[DetResult]
) -> PrecRecTable:
    confs = np.concat([r.confs for r in results])
    tp_pred_flags = np.concat([r.tp_pred_flags for r in results])
    fn_gt_flags = np.concat([r.fn_gt_flags for r in results])
    num_preds = len(tp_pred_flags)

    sort_indice = np.argsort(confs)[::-1]
    confs = confs[sort_indice]
    tp_pred_flags = tp_pred_flags[sort_indice]
    fn_gt_flags = fn_gt_flags[sort_indice]

    pred_counts = np.arange(num_preds) + 1
    cum_tps = np.cumsum(tp_pred_flags)
    precs = cum_tps / pred_counts
    recs = cum_tps / pred_counts

    prec_rec_table = PrecRecTable(confs, cum_tps, precs, recs)

    return prec_rec_table
    

def get_ap_prCurve(
    prec_rec_table: PrecRecTable
) -> Tuple[float]:
    recs = prec_rec_table.recs
    precs = prec_rec_table.precs

    # COCO-style interpolation
    recs = np.concatenate([[0], recs, [1]])
    precs = np.concatenate([[0], precs, [0]])
    precs = np.maximum.accumulate(precs[::-1])[::-1]
    
    ap = np.trapezoid(precs, recs).item()

    return ap, 

def get_