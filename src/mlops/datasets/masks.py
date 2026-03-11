from typing import List, Union, Literal
import pycocotools.mask as pycocomask

import cv2
import numpy as np
from numpy.typing import NDArray

from mlops.datasets.types import MaskArrType, MasksArrType
from mlops.datasets.types import RleType
from mlops.datasets.types import PolyArrType, PolyLabelmeType, PolyCocoType, PolyYoloType


def convert_maskArr2rle(
    masks: Union[MaskArrType, MasksArrType]
) -> List[RleType]:
    if masks.ndim == 2:
        masks = np.asfortranarray(masks)
        rles = pycocomask.encode(masks)
        rles = [rles]
    else:
        masks = np.transpose(masks, (1, 2, 0))
        masks = np.asfortranarray(masks)
        rles = pycocomask.encode(masks)
    return rles

def convert_maskArr2polyArr(
    mask: MaskArrType,
    approx_flag: bool,
) -> List[PolyArrType]:
    """
    Only retrieve the outer contours.
    """
    mask = mask.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys = []
    for cnt in cnts:
        if approx_flag:
            eps = 0.001 * cv2.arcLength(cnt, True)
            cnt_approx = cv2.approxPolyDP(cnt, eps, True)
        else:
            cnt_approx = cnt
        
        poly = np.squeeze(cnt_approx, axis = 1)
        polys.append(poly)
    
    return polys

def convert_maskArr2polyLabelme(
    mask: MaskArrType,
    approx_flag: bool,
) -> List[PolyLabelmeType]:
    polys = convert_maskArr2polyArr(mask, approx_flag)
    polys = [p.tolist() for p in polys]
    return polys

def convert_maskArr2polyCoco(
    mask: MaskArrType,
    approx_flag: bool
) -> List[PolyCocoType]:
    polys = convert_maskArr2polyArr(mask, approx_flag)
    polys = [p.flatten().tolist() for p in polys]
    return polys

def convert_maskArr2polyYolo(
    mask: MaskArrType,
    approx_flag: bool
) -> List[PolyYoloType]:
    img_h, img_w = mask.shape
    polys = convert_maskArr2polyArr(mask, approx_flag)
    polys_coco = []

    for poly in polys:
        poly_coco = poly.copy()
        poly_coco[:, 0] = poly_coco[:, 0] / img_h
        poly_coco[:, 1] = poly_coco[:, 1] / img_w
        polys_coco.append(poly_coco)
    
    return polys_coco

def concat_masks(
    masks: List[Union[MaskArrType], List[MasksArrType]]
) -> MasksArrType:
    masks_ = []
    for m in masks:
        if m.ndim == 2:
            m = m[None, ...]
        masks_.append(m)
    masks = np.concat(masks_, axis = 0)
    return masks

def ious_masks(
    masksA: MasksArrType,
    masksB: MasksArrType,
    mode: Literal["iou", "iof"]
) -> NDArray[np.floating]:
    """
    Intro
    -----
    iou: area(intersect) / area(union)
    iof: area(intersect) / area(masksA)

    Returns
    -----
    - `ious or iofs, NDArray[np.floating], (numA, numB)`
    """
    assert mode in ["iou", "iof"]

    masksA = masksA.astype(np.bool_)
    masksB = masksB.astype(np.bool_)

    masksA_area = np.sum(masksA, axis = (1, 2))[None, :]
    masksB_area = np.sum(masksB, axis = (1, 2))[:, None]

    inter = np.logical_and(
        masksA[None, :, :, :], 
        masksB[:, None, :, :],
    )
    inter_area = np.sum(inter, axis = (2, 3))

    if mode == "iou":
        union_area = masksA_area + masksB_area - inter_area
        ious = inter_area / (union_area + 1e-8)
        return ious
    else:
        iofs = inter_area / (masksA_area + 1e-8)
        return iofs