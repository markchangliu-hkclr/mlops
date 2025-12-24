from mlops.shapes.typing import BBoxesXYWHType, \
    BBoxesXYXYType, BBoxesLabelmeType


__all__ = [
    "bboxes2bboxes_xyxy2xywh",
    "bboxes2bboxes_xywh2xyxy",
    "bboxes2bboxes_xyxy2labelme"
]


def bboxes2bboxes_xyxy2xywh(
    bboxes_xyxy: BBoxesXYXYType
) -> BBoxesXYWHType:
    bboxes_xywh = bboxes_xyxy.copy()
    bboxes_xywh[:, [2, 3]] = bboxes_xyxy[:, [2, 3]] - bboxes_xyxy[:, [0, 1]]
    return bboxes_xywh

def bboxes2bboxes_xywh2xyxy(
    bboxes_xywh: BBoxesXYWHType
) -> BBoxesXYXYType:
    bboxes_xyxy = bboxes_xywh.copy()
    bboxes_xyxy[:, [2, 3]] = bboxes_xywh[:, [0, 1]] + bboxes_xywh[:, [2, 3]]
    return bboxes_xyxy

def bboxes2bboxes_xyxy2labelme(
    bboxes_xyxy: BBoxesXYXYType
) -> BBoxesLabelmeType:
    bboxes_labelme = bboxes_xyxy.copy().reshape(-1, 2, 2).tolist()
    return bboxes_labelme