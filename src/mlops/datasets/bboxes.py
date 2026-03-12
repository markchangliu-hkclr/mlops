from typing import Union, List, Tuple, Literal

import numpy as np
from numpy.typing import NDArray

from mlops.datasets.types import (
    BBoxArrType, BBoxesArrType, BBoxCocoType, 
    BBoxLabelmeType, BBoxYoloType, BBoxType, BBoxesType, BBoxFormat
)


__all__ = [
    "convert_bbox_format",
    "concat_bbox",
    "iou_bbox"
]


def _bboxArr2bboxLabelme(
    bbox: BBoxArrType
) -> BBoxLabelmeType:
    bbox = bbox.reshape(2, 2).tolist()
    return bbox

def _bboxesArr2bboxesLabelme(
    bboxes: BBoxesArrType
) -> List[BBoxLabelmeType]:
    bboxes = bboxes.copy().reshape(-1, 2, 2).tolist()
    return bboxes

def _bboxArr2bboxCoco(
    bbox: BBoxArrType
) -> BBoxCocoType:
    bbox = bbox.tolist()
    x1, y1, x2, y2 = bbox
    bbox = [x1, y1, x2 - x1, y2 - y1]
    return bbox

def _bboxesArr2bboxesCoco(
    bboxes: BBoxesArrType
) -> List[BBoxCocoType]:
    bboxes[:, [2, 3]] = bboxes[:, [2, 3]] - bboxes[:, [0, 1]]
    bboxes = bboxes.tolist()
    return bboxes

def _bboxArr2bboxYolo(
    bbox: BBoxArrType,
    img_hw: Tuple[int, int]
) -> List[BBoxYoloType]:
    img_h, img_w = img_hw
    x1, y1, x2, y2 = bbox
    w = x2 - x1 
    h = y2 - y1
    x_ctr_norm = (x2 + x1) / img_w
    y_ctr_norm = (y2 + y1) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    bbox = [x_ctr_norm, y_ctr_norm, w_norm, h_norm]
    return bbox

def _bboxesArr2bboxesYolo(
    bboxes: BBoxesArrType,
    img_hw: Tuple[int, int]
) -> List[BBoxYoloType]:
    img_h, img_w = img_hw
    bboxes[:, [2, 3]] = bboxes[:, [2, 3]] - bboxes[:, [0, 1]]
    bboxes[:, [0, 1]] = bboxes[:, [0, 1]] + 0.5 * bboxes[:, [2, 3]]
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / img_w
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / img_h
    bboxes = bboxes.tolist()
    return bboxes

def _bboxLabelme2bboxArr(
    bboxLabelme: BBoxLabelmeType
) -> BBoxArrType:
    bbox = np.asarray(bboxLabelme)
    return bbox

def _bboxesLabelme2bboxesArr(
    bboxLabelme: List[BBoxLabelmeType]
) -> BBoxesArrType:
    bboxes = np.asarray(bboxLabelme)
    return bboxes

def _bboxLabelme2bboxCoco(
    bbox: BBoxLabelmeType
) -> BBoxCocoType:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    bbox = [bbox[0], bbox[1], w, h]
    return bbox

def _bboxesLabelme2bboxesCoco(
    bboxes: List[BBoxLabelmeType]
) -> List[BBoxCocoType]:
    bboxes = [_bboxLabelme2bboxCoco(b) for b in bboxes]
    return bboxes

def _bboxLabelme2bboxYolo(
    bbox: BBoxLabelmeType,
    img_hw: Tuple[int, int]
) -> BBoxYoloType:
    x_ctr = (bbox[2] + bbox[0]) / 2 / img_hw[1]
    y_ctr = (bbox[3] + bbox[1]) / 2 / img_hw[0]
    w = (bbox[2] - bbox[0]) / img_hw[1]
    h = (bbox[3] - bbox[1]) / img_hw[0]
    bbox = [x_ctr, y_ctr, w, h]
    return bbox

def _bboxesLabelme2bboxesYolo(
    bboxes: List[BBoxLabelmeType],
    img_hw: Tuple[int, int]
) -> List[BBoxYoloType]:
    bboxes = [_bboxLabelme2bboxYolo(b, img_hw) for b in bboxes]
    return bboxes

def _bboxCoco2bboxArr(
    bboxCoco: BBoxCocoType
) -> BBoxArrType:
    bboxes = np.asarray(bboxCoco)
    bboxes[2, 3] = bboxes[0, 1] + bboxes[2, 3]
    return bboxes

def _bboxesCoco2bboxesArr(
    bboxes: List[BBoxCocoType]
) -> BBoxesArrType:
    bboxes = np.asarray(bboxes)
    bboxes[:, [2, 3]] = bboxes[:, [0, 1]] + bboxes[:, [2, 3]]
    return bboxes

def _bboxCoco2bboxLabelme(
    bbox: BBoxCocoType
) -> BBoxLabelmeType:
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    bbox = [x1, y1, x2, y2]
    return bbox

def _bboxesCoco2bboxesLabelme(
    bboxes: List[BBoxCocoType]
) -> List[BBoxLabelmeType]:
    bboxes = np.asarray(bboxes).reshape(-1, 4)
    bboxes[:, [2, 3]] = bboxes[:, [0, 1]] + bboxes[:, [2, 3]]
    bboxes = bboxes.tolist()
    return bboxes

def _bboxCoco2bboxYolo(
    bbox: BBoxCocoType,
    img_hw: Tuple[int, int]
) -> BBoxYoloType:
    x1, y1, w, h = bbox
    x_ctr_norm = (x1 + w + x1) / 2 / img_hw[1]
    y_ctr_norm = (y1 + h + y1) / 2 / img_hw[0]
    w_norm = w / img_hw[1]
    h_nrom = h / img_hw[0]
    bbox = [x_ctr_norm, y_ctr_norm, w_norm, h_nrom]
    return bbox

def _bboxesCoco2bboxesYolo(
    bboxes: List[BBoxCocoType],
    img_hw: Tuple[int, int]
) -> List[BBoxYoloType]:
    bboxes = np.asarray(bboxes).reshape(-1, 4)
    bboxes[:, 0] = (bboxes[:, 0] + bboxes[:, 2] + bboxes[:, 0]) / 2 / img_hw[1]
    bboxes[:, 1] = (bboxes[:, 1] + bboxes[:, 3] + bboxes[:, 1]) / 2 / img_hw[0]
    bboxes[:, 2] = bboxes[:, 2] / img_hw[1]
    bboxes[:, 3] = bboxes[:, 3] / img_hw[0]
    bboxes = bboxes.tolist()
    return bboxes

def _bboxYolo2bboxArr(
    bboxYolo: BBoxYoloType,
    img_hw: Tuple[int, int]
) -> BBoxArrType:
    img_h, img_w = img_hw
    bboxes = np.asarray(bboxYolo)
    bboxes[0, 2] = bboxes[0, 2] * img_w
    bboxes[1, 3] = bboxes[1, 3] * img_h
    bboxes[0, 1] = bboxes[0, 1] - 0.5 * bboxes[2, 3]
    bboxes[2, 3] = bboxes[0, 1] + bboxes[2, 3]
    bboxes = bboxes
    return bboxes

def _bboxesYolo2bboxesArr(
    bboxesYolo: List[BBoxYoloType], 
    img_hw: Tuple[int, int]
) -> BBoxesArrType:
    img_h, img_w = img_hw
    bboxes = np.asarray(bboxesYolo)
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * img_w
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * img_h
    bboxes[:, [0, 1]] = bboxes[:, [0, 1]] - 0.5 * bboxes[:, [2, 3]]
    bboxes[:, [2, 3]] = bboxes[:, [0, 1]] + bboxes[:, [2, 3]]
    bboxes = bboxes
    return bboxes

def _bboxYolo2bboxLabelme(
    bbox: BBoxYoloType,
    img_hw: Tuple[int, int]
) -> BBoxLabelmeType:
    x_ctr_norm, y_ctr_norm, w_norm, h_norm = bbox
    x1 = (x_ctr_norm - w_norm / 2) * img_hw[1]
    x2 = (x_ctr_norm + w_norm / 2) * img_hw[1]
    y1 = (y_ctr_norm - h_norm / 2) * img_hw[0]
    y2 = (y_ctr_norm + h_norm / 2) * img_hw[0]
    bbox = [x1, y1, x2, y2]
    return bbox

def _bboxesYolo2bboxesLabelme(
    bboxes: List[BBoxYoloType],
    img_hw: Tuple[int, int]
) -> List[BBoxLabelmeType]:
    xs_ctr_norm = bboxes[:, 0]
    ys_ctr_norm = bboxes[:, 1]
    ws_norm = bboxes[:, 2]
    hs_norm = bboxes[:, 3]
    x1s = (xs_ctr_norm - ws_norm / 2) * img_hw[1]
    x2s = (xs_ctr_norm + ws_norm / 2) * img_hw[1]
    y1s = (ys_ctr_norm - hs_norm / 2) * img_hw[0]
    y2s = (ys_ctr_norm + hs_norm / 2) * img_hw[0]
    bboxes[:, 0] = x1s
    bboxes[:, 1] = y1s
    bboxes[:, 2] = x2s
    bboxes[:, 3] = y2s
    bboxes = bboxes.tolist()
    return bboxes

def _bboxYolo2bboxCoco(
    bboxYolo: BBoxYoloType,
    img_hw: Tuple[int, int]
) -> BBoxCocoType:
    x_ctr_norm, y_ctr_norm, w_norm, h_norm = bboxYolo
    x1 = (x_ctr_norm - w_norm / 2) * img_hw[1]
    y1 = (y_ctr_norm + h_norm / 2) * img_hw[0]
    w = w_norm * img_hw[1]
    h = h_norm * img_hw[0]
    bbox = [x1, y1, w, h]
    return bbox

def _bboxesYolo2bboxesCoco(
    bboxes: List[BBoxYoloType],
    img_hw: Tuple[int, int]
) -> List[BBoxCocoType]:
    bboxes = np.asarray(bboxes).reshape(-1, 4)
    xs_ctr_norm = bboxes[:, 0]
    ys_ctr_norm = bboxes[:, 1]
    ws_norm = bboxes[:, 2]
    hs_norm = bboxes[:, 3]
    x1s = (xs_ctr_norm - 0.5 * ws_norm) * img_hw[1]
    y1s = (ys_ctr_norm - 0.5 * hs_norm) * img_hw[0]
    ws = ws_norm * img_hw[1]
    hs = hs_norm * img_hw[0]
    bboxes[:, 0] = x1s
    bboxes[:, 1] = y1s
    bboxes[:, 2] = ws
    bboxes[:, 3] = hs
    bboxes = bboxes.tolist()
    return bboxes

def convert_bbox_format(
    bbox: Union[BBoxType, BBoxesType],
    img_hw: Tuple[int, int],
    src_format: BBoxFormat,
    dst_format: BBoxFormat,
    bbox_count: Literal["single", "multi"]
) -> Union[BBoxType, BBoxesType]:
    assert src_format == dst_format
    assert bbox_count in ["single", "multi"]

    if src_format == "arr" and dst_format == "labelme":
        if bbox_count == "single":
            bbox = _bboxArr2bboxLabelme(bbox)
        else:
            bbox = _bboxesArr2bboxesLabelme(bbox)
    elif src_format == "arr" and dst_format == "coco":
        if bbox_count == "single":
            bbox = _bboxArr2bboxCoco(bbox)
        else:
            bbox = _bboxesArr2bboxesCoco(bbox)
    elif src_format == "arr" and dst_format == "yolo":
        if bbox_count == "single":
            bbox = _bboxArr2bboxYolo(bbox, img_hw)
        else:
            bbox = _bboxesArr2bboxesYolo(bbox, img_hw)
    elif src_format == "labelme" and dst_format == "arr":
        if bbox_count == "single":
            bbox = _bboxLabelme2bboxArr(bbox)
        else:
            bbox = _bboxesLabelme2bboxesArr(bbox)
    elif src_format == "labelme" and dst_format == "coco":
        if bbox_count == "single":
            bbox = _bboxLabelme2bboxCoco(bbox)
        else:
            bbox = _bboxesLabelme2bboxesCoco(bbox)
    elif src_format == "labelme" and dst_format == "yolo":
        if bbox_count == "single":
            bbox = _bboxLabelme2bboxYolo(bbox, img_hw)
        else:
            bbox = _bboxesLabelme2bboxesYolo(bbox, img_hw)
    elif src_format == "coco" and dst_format == "arr":
        if bbox_count == "single":
            bbox = _bboxCoco2bboxArr(bbox)
        else:
            bbox = _bboxesCoco2bboxesArr(bbox)
    elif src_format == "coco" and dst_format == "labelme":
        if bbox_count == "single":
            bbox = _bboxCoco2bboxLabelme(bbox)
        else:
            bbox = _bboxesCoco2bboxesLabelme(bbox)
    elif src_format == "coco" and dst_format == "yolo":
        if bbox_count == "single":
            bbox = _bboxCoco2bboxYolo(bbox, img_hw)
        else:
            bbox = _bboxesCoco2bboxesYolo(bbox, img_hw)
    elif src_format == "yolo" and dst_format == "arr":
        if bbox_count == "single":
            bbox = _bboxYolo2bboxArr(bbox, img_hw)
        else:
            bbox = _bboxesYolo2bboxesArr(bbox, img_hw)
    elif src_format == "yolo" and dst_format == "coco":
        if bbox_count == "single":
            bbox = _bboxYolo2bboxCoco(bbox, img_hw)
        else:
            bbox = _bboxesYolo2bboxesCoco(bbox, img_hw)
    elif src_format == "yolo" and dst_format == "labelme":
        if bbox_count == "single":
            bbox = _bboxYolo2bboxLabelme(bbox, img_hw)
        else:
            bbox = _bboxesYolo2bboxesLabelme(bbox, img_hw)
    else:
        raise NotImplementedError
    
    return bbox

def concat_bbox(
    bboxes_list: List[Union[BBoxArrType, BBoxesArrType]]
) -> BBoxesArrType:
    for i in range(len(bboxes_list)):
        if len(bboxes_list[i].shape) == 1:
            bboxes_list[i] = bboxes_list[i].reshape(-1, 4)
    
    bboxes = np.concat(bboxes_list, axis = 0)

    return bboxes

def iou_bbox(
    bboxesA: Union[BBoxesArrType, BBoxArrType],
    bboxesB: Union[BBoxesArrType, BBoxArrType],
    mode: Literal["iou", "iof"]
) -> NDArray[np.floating]:
    """
    Intro
    -----
    iou: area(intersect) / area(union)
    iof: area(intersect) / area(bboxesA)

    Args
    -----
    - `bboxesA`: `(numA, 4)`,
    - `bboxesB`: `(numB, 4)`

    Returns
    -----
    - `ious or iofs`: `(numA, numB)`
    """
    assert mode in ["iou", "iof"]

    bboxesA = bboxesA.copy().reshape(-1, 4)
    bboxesB = bboxesB.copy().reshape(-1, 4)

    # Expand dimensions to enable broadcasting
    # bboxesA: (num_bboxesA, 1, 4)
    # bboxesB: (1, num_bboxesB, 4)
    bboxesA = bboxesA[:, None, :]  # shape (num_bboxesA, 1, 4)
    bboxesB = bboxesB[None, :, :]  # shape (1, num_bboxesB, 4)
    
    # Compute intersection coordinates
    # (num_bboxesA, num_bboxesB)
    x1_inter = np.maximum(bboxesA[..., 0], bboxesB[..., 0])
    y1_inter = np.maximum(bboxesA[..., 1], bboxesB[..., 1])
    x2_inter = np.minimum(bboxesA[..., 2], bboxesB[..., 2])
    y2_inter = np.minimum(bboxesA[..., 3], bboxesB[..., 3])
    
    # Compute intersection area
    w_inter = np.maximum(0, x2_inter - x1_inter)
    h_inter = np.maximum(0, y2_inter - y1_inter)
    area_inter = w_inter * h_inter

    if mode == "iof":
        areasA = np.prod(bboxesA[:, [2, 3]] - bboxesA[:, [0, 1]], axis = 1)
        iofs = area_inter / (areasA[:, None] + 1e-8)
        return iofs
    else:
        # Compute union area
        x1_union = np.minimum(bboxesA[..., 0], bboxesB[..., 0])
        y1_union = np.minimum(bboxesA[..., 1], bboxesB[..., 1])
        x2_union = np.maximum(bboxesA[..., 2], bboxesB[..., 2])
        y2_union = np.maximum(bboxesA[..., 3], bboxesB[..., 3])

        w_union = np.maximum(0, x2_union - x1_union)
        h_union = np.maximum(0, y2_union - y1_union)
        area_union = w_union * h_union

        ious = area_inter / (area_union + 1e-8)
        return ious

