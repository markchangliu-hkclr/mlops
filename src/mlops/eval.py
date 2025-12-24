import os
from typing import TypedDict, List, TypeAlias, Dict, Literal, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from mlops.models.base import BaseModel


class MatchResultType(TypedDict):
    """
    - `dt_match_flags`: `(num_dts, )`, `bool`
    - `dt_match_gt_idx`: `(num_dts, )`, `gt_idx` or `-1`
    - `gt_match_flags`: `(num_gts, )`, `bool`
    - `gt_match_dt_idx`: `(num_gts, (num_match_dts, ))`, sorted by ious
    """
    dt_match_flags: NDArray[np.bool]
    dt_match_gt_idx: NDArray[np.integer]
    gt_match_flags: NDArray[np.bool]
    gt_match_dt_idx: List[List[int]]

class MatchResultSummaryType(TypedDict):
    """
    - `num_tps`: `int`
    - `num_fps`: `int`
    - `num_fns`: `int`
    - `num_dts`: `int`
    - `num_gts`: `int`
    """
    num_tps: int
    num_fps: int
    num_fns: int
    num_dts: int
    num_gts: int

class MetricSingleClsType(TypedDict):
    """
    - `prec`: `List[float]`, prec at different ious
    - `rec`: `List[float]`, rec at different ious
    - `ap`: `List[float]`, ap at different ious
    - `ap_overall`: float
    """

class MetricMultiClsType(TypedDict):
    metric_ious: Dict[str, MetricSingleClsType]
    map: List[float]
    map_overall: List[float]


def match_dt_gt(
    ious: NDArray[np.number],
    iou_thres: float,
    gt_match_multi_dt_flag: bool
) -> MatchResultType:
    """
    Params
    -----
    - `ious`: `(num_dts, num_gts)`
    - `iou_thres`: `float`
    - `multiple_dt_match_flag`: `bool`
    """
    ious_copy = np.copy(ious)

    num_dts, num_gts = ious.shape

    match_res: MatchResultType = {
        "dt_match_flags": np.zeros(num_dts).astype(np.bool),
        "dt_match_gt_idx": np.ones(num_dts) * -1,
        "gt_match_flags": np.zeros(num_gts).astype(np.bool),
        "gt_match_dt_idx": [[] for i in range(num_gts)]
    }

    num_dts_remain = num_dts

    for i in range(num_gts):
        if num_dts_remain == 0:
            break

        iou_max_idx = np.argmax(ious_copy)
        dt_idx, gt_idx = np.unravel_index(iou_max_idx, ious_copy.shape)
        dt_idx = dt_idx.item()
        gt_idx = gt_idx.item()
        max_iou = ious_copy[dt_idx, gt_idx]

        if max_iou < iou_thres:
            break

        match_res["dt_match_flags"][dt_idx] = True
        match_res["dt_match_gt_idx"][dt_idx] = gt_idx
        match_res["gt_match_flags"][gt_idx] = True
        match_res["gt_match_dt_idx"][gt_idx].append(dt_idx)

        ious_copy[dt_idx, :] = float("-inf")
        ious_copy[:, gt_idx] = float("-inf")

        num_dts_remain -= 1
    
    if not gt_match_multi_dt_flag:
        return match_res
    
    ious_remain = np.copy(ious)
    ious_remain[match_res["dt_match_flags"], :] = float("-inf")

    for i in range(num_dts_remain):
        iou_max_idx = np.argmax(ious_remain)
        dt_idx, gt_idx = np.unravel_index(iou_max_idx, ious_remain.shape)
        dt_idx = dt_idx.item()
        gt_idx = gt_idx.item()
        max_iou = ious_remain[dt_idx, gt_idx]

        if max_iou < iou_thres:
            break

        match_res["dt_match_flags"][dt_idx] = True
        match_res["dt_match_gt_idx"][dt_idx] = gt_idx
        match_res["gt_match_dt_idx"][gt_idx].append(dt_idx)

        ious_remain[dt_idx, :] = float("-inf")
        ious_remain[:, gt_idx] = float("-inf")
    
    return match_res

def get_match_summary(
    match_res: MatchResultType
) -> MatchResultSummaryType:
    summary: MatchResultSummaryType = {
        "num_dts": 0,
        "num_gts": 0,
        "num_tps": 0,
        "num_fps": 0,
        "num_fns": 0,
    }

    num_dts = len(match_res["dt_match_flags"])
    num_gts = len(match_res["gt_match_flags"])
    num_tps = np.sum(match_res["dt_match_flags"]).item()

    summary["num_gts"] = len(match_res["gt_match_flags"])
    summary["num_dts"] = len(match_res["dt_match_flags"])
    summary["num_fps"] = np.sum(match_res)
    pass

def eval_img(
    img_p: str,
    model: object,
    export_vis_flag: bool,
    export_vis_p: str,
    export_img_res_flag: bool,
    export_img_res_p: str,
) -> Tuple[MatchResultSummaryType, MetricType]:
    pass

def eval_labelme_dataset_online(
    dataset_root: str,
    model: object,
    metric_avg_mode: Literal["img", "overall"],
    export_vis_flag: bool,
    export_vis_root: str,
    export_img_res_flag: bool,
    export_img_res_root: str,
) -> MetricType:
    pass
    

def test1() -> None:
    ious = [
        [0.06374037, 0.07814641, 0.06835990],
        [0.00127131, 0.82855156, 0.09722756],
        # [0.69328143, 0.2729694 , 0.06887056],
        # [0.90708562, 0.59382972, 0.0870394 ],
        # [0.50794111, 0.89787563, 0.14130168]
    ]

    ious = np.asarray(ious)

    match_res = match_dt_gt(
        ious, 0.5, False
    )

    for k, v in match_res.items():
        print(k)
        print(v)
        print()


if __name__ == "__main__":
    test1()
