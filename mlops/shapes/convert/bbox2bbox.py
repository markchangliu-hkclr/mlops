from typing import Tuple

import numpy as np

import mlops.shapes.typedef.bboxes as bbox_type


def bboxes2bboxes_xyxy2xywh(
    bboxes_xyxy: bbox_type.BBoxesXYXYArrType
) -> bbox_type.BBoxesXYWHArrType:
    bboxes_xywh = bboxes_xyxy.copy()
    bboxes_xywh[:, [2, 3]] = bboxes_xyxy[:, [2, 3]] - bboxes_xyxy[:, [0, 1]]
    return bboxes_xywh

def bboxes2bboxes_xywh2xyxy(
    bboxes_xywh: bbox_type.BBoxesXYWHArrType
) -> bbox_type.BBoxesXYXYArrType:
    bboxes_xyxy = bboxes_xywh.copy()
    bboxes_xyxy[:, [2, 3]] = bboxes_xywh[:, [0, 1]] + bboxes_xywh[:, [2, 3]]
    return bboxes_xyxy

def bboxes2bboxes_xyxy2labelme(
    bboxes_xyxy: bbox_type.BBoxesXYXYArrType
) -> bbox_type.BBoxesLabelmeType:
    bboxes_labelme = bboxes_xyxy.copy().reshape(-1, 2, 2).tolist()
    return bboxes_labelme

def bbox2bbox_yolo2labelme(
    bbox: bbox_type.BBoxYoloType,
    img_hw: Tuple[int, int]
) -> bbox_type.BBoxLabelmeType:
    x_ctr_norm = bbox[0]
    y_ctr_norm = bbox[1]
    w_norm = bbox[2]
    h_norm = bbox[3]

    x1 = int((x_ctr_norm - 0.5 * w_norm) * img_hw[1])
    x2 = int((x_ctr_norm + 0.5 * w_norm) * img_hw[1])
    y1 = int((y_ctr_norm - 0.5 * h_norm) * img_hw[0])
    y2 = int((y_ctr_norm + 0.5 * h_norm) * img_hw[0])
    bbox = [[x1, y1], [x2, y2]]
    return bbox