from typing import Tuple, Union, List

import numpy as np
from numpy.typing import NDArray


__all__ = [
    "match_1vs1",
    "match_precision_focus",
    "match_recall_focus",
    "get_tp_fp_fn"
]


def match_1vs1(
    scores: NDArray[np.floating],
    thres: float,
) -> Tuple[NDArray[np.integer], NDArray[np.integer]]:
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
) -> Tuple[NDArray[np.integer], NDArray[np.bool_]]:
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
) -> Tuple[NDArray[np.bool_], NDArray[np.integer]]:
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

def get_tp_fn(
    pred_matches: Union[NDArray[np.integer], NDArray[np.bool_]],
    gt_matches: Union[NDArray[np.integer], NDArray[np.bool_]],
) -> Tuple[List[int], List[int], List[int]]:
    """
    Args
    -----
    - `pred_matches`
        - `NDArray[np.integer]: (num_preds, )`
        - `NDArray[np.bool_]: (num_preds, num_gts)`
    - `gt_matches`
        - `NDArray[np.integer]: (num_gts, )`
        - `NDArray[np.bool_]: (num_gts, num_preds)`

    Returns
    -----
    - `tp_pred_flags: NDArray[np.bool_], (num_preds, )`
    - `fn_gt_flags: NDArray[np.bool_], (num_gts, )`
    """
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
