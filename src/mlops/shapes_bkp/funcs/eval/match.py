from typing import Tuple, Union, List, Literal

import numpy as np
from numpy.typing import NDArray

from mlops.shapes.typing.eval import PredMatchResType, GTMatchResType, \
    PredMatchRes1DType, PredMatchRes2DType, GTMatchRes1DType, GTMatchRes2DType


__all__ = [
    "match",
    "match_1vs1",
    "match_precision_focus",
    "match_recall_focus",
    "get_tp_fp_fn"
]


def match(
    scores: NDArray[np.floating],
    thres: float,
    match_mode: Literal["1vs1", "prec_focus", "rec_focus"]
) -> Tuple[PredMatchResType, GTMatchResType]:
    assert match_mode in ["1vs1", "prec_focus", "rec_focus"]

    if match_mode == "1vs1":
        pred_matches, gt_matches = match_1vs1(scores, thres)
    elif match_mode == "prec_focus":
        pred_matches, gt_matches = match_prec_focus(scores, thres)
    else:
        pred_matches, gt_matches = match_rec_focus(scores, thres)
    
    return pred_matches, gt_matches


def match_1vs1(
    scores: NDArray[np.floating],
    thres: float,
) -> Tuple[PredMatchRes1DType, GTMatchRes1DType]:
    """
    One GT match at most has 1 PRED, one PRED match at most 1 GT
    - PRED only match to highest-score and >threshold GT
    - if this GT is already matched, this PRED is FP

    Args
    -----
    - `scores: NDArray[np.floating], (num_gts, num_preds)`
    - `thres: float`

    Returns
    -----
    - `pred_matches: NDArray[np.integer], (num_preds, )`, 
    matched gt ids, -1 means no match
    - `gt_matches: NDArray[np.integer], (num_gts, )`, 
    matched pred ids, -1 means no match
    """
    num_gts, num_preds = scores.shape
    pred_matches = np.ones(num_preds, dtype = np.int32) * -1
    gt_matches = np.ones(num_gts, dtype = np.int32) * -1

    for pred_id in range(num_preds):
        max_score_gt_id = np.argmax(scores[:, pred_id])
        max_score = scores[:, pred_id][max_score_gt_id]

        if not max_score > thres:
            continue

        if gt_matches[max_score_gt_id] >= 0:
            continue

        gt_matches[max_score_gt_id] = pred_id
        pred_matches[pred_id] = max_score_gt_id

    return pred_matches, gt_matches

def match_prec_focus(
    scores: NDArray[np.floating],
    thres: float,
) -> Tuple[PredMatchRes1DType, GTMatchRes2DType]:
    """
    GT can match to multiple PREDs
    - PRED match to highest-score and >threshold GTs
    - GT may be repetitively matched

    Args
    -----
    - `scores: NDArray[np.floating], (num_gts, num_preds)`
    - `thres: float`

    Returns
    -----
    - `pred_matches: NDArray[np.integer], (num_preds, )`, 
    matched gt ids, -1 means no match
    - `gt_matches: NDArray[np.bool], (num_gts, num_preds)`, 
    """
    num_gts, num_preds = scores.shape
    pred_matches = np.ones(num_preds, dtype = np.int32) * -1
    gt_matches = np.zeros((num_gts, num_preds), dtype = np.bool_)

    for pred_id in range(num_preds):
        max_score_gt_id = np.argmax(scores[:, pred_id])
        max_score = scores[:, pred_id][max_score_gt_id]

        if not max_score > thres:
            continue

        gt_matches[max_score_gt_id, pred_id] = True
        pred_matches[pred_id] = max_score_gt_id

    return pred_matches, gt_matches

def match_rec_focus(
    scores: NDArray[np.floating],
    thres: float,
) -> Tuple[PredMatchRes2DType, GTMatchRes1DType]:
    """
    PRED can match to multiple GTs
    - GT match to highest-score and >threshold PRED
    - PRED may be repetitively matched

    Args
    -----
    - `scores: NDArray[np.floating], (num_gts, num_preds)`
    - `thres: float`

    Returns
    -----
    - `pred_matches: NDArray[np.bool_], (num_preds, num_gts)`, 
    - `gt_matches: NDArray[np.integer], (num_gts, )`, 
    """
    num_gts, num_preds = scores.shape
    pred_matches = np.zeros((num_preds, num_gts), dtype = np.bool_)
    gt_matches = np.ones(num_gts, dtype = np.int32) * -1

    for gt_id in range(num_gts):
        max_score_pred_id = np.argmax(scores[gt_id, :])
        max_score = scores[gt_id, :][max_score_pred_id]

        if not max_score > thres:
            continue

        gt_matches[gt_id] = max_score_pred_id
        pred_matches[max_score_pred_id, gt_id] = True

    return pred_matches, gt_matches


