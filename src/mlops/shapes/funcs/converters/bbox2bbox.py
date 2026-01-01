from ...typing import (
    BBoxesArrXYWHType,
    BBoxesArrXYXYType,
    BBoxesLabelmeType
)


__all__ = [
    "bboxesArrXYXY_to_bboxesArrXYWH",
    "bboxesArrXYWH_to_bboxesArrXYXY",
    "bboxesArrXYXY_to_bboxesLabelme"
]


def bboxesArrXYXY_to_bboxesArrXYWH(
    bboxes_xyxy: BBoxesArrXYXYType
) -> BBoxesArrXYWHType:
    bboxes_xywh = bboxes_xyxy.copy()
    bboxes_xywh[:, [2, 3]] = bboxes_xyxy[:, [2, 3]] - bboxes_xyxy[:, [0, 1]]
    return bboxes_xywh

def bboxesArrXYWH_to_bboxesArrXYXY(
    bboxes_xywh: BBoxesArrXYWHType
) -> BBoxesArrXYXYType:
    bboxes_xyxy = bboxes_xywh.copy()
    bboxes_xyxy[:, [2, 3]] = bboxes_xywh[:, [0, 1]] + bboxes_xywh[:, [2, 3]]
    return bboxes_xyxy

def bboxesArrXYXY_to_bboxesLabelme(
    bboxes_xyxy: BBoxesArrXYXYType
) -> BBoxesLabelmeType:
    bboxes_labelme = bboxes_xyxy.copy().reshape(-1, 2, 2).tolist()
    return bboxes_labelme