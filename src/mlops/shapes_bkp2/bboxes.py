"""
# Bounding Boxes Module

This module provides functions for converting and processing bounding bboxes.

The module is mainly used in dataset preparations and evaluations.

## Funcs

### Convert
- `convert_bboxArr2bboxLabelme`
- `convert_bboxArr2bboxCoco`
- `convert_bboxArr2bboxYolo`
- `convert_bboxLabelme2bboxArr`
- `convert_bboxCoco2bboxArr`
- `convert_bboxYolo2bboxArr`

### Concat
- `concat_bboxes`

### IoU
- `iou_bboxes`

### Visualize
- `draw_bboxes`
"""

from typing import Union, List, Tuple, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from mlops.shapes.types import (
    BBoxArrType, BBoxesArrType, BBoxCocoType, 
    BBoxLabelmeType, BBoxYoloType
)


def convert_bboxArr2bboxLabelme(
    bboxArr: Union[BBoxArrType, BBoxesArrType]
) -> List[BBoxLabelmeType]:
    bboxes = bboxes.copy().reshape(-1, 2, 2).tolist()

    return bboxes

def convert_bboxArr2bboxCoco(
    bboxArr: Union[BBoxArrType, BBoxesArrType]
) -> List[BBoxCocoType]:
    bboxes = bboxArr.copy().reshape(-1, 4)
    bboxes[:, [2, 3]] = bboxes[:, [2, 3]] - bboxes[:, [0, 1]]
    bboxes = bboxes.tolist()

    return bboxes

def convert_bboxArr2bboxYolo(
    bboxArr: Union[BBoxArrType, BBoxesArrType],
    img_hw: Tuple[int, int]
) -> List[BBoxYoloType]:
    img_h, img_w = img_hw
    bboxes = bboxArr.copy().reshape(-1, 4)
    bboxes[:, [2, 3]] = bboxes[:, [2, 3]] - bboxes[:, [0, 1]]
    bboxes[:, [0, 1]] = bboxes[:, [0, 1]] + 0.5 * bboxes[:, [2, 3]]
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / img_w
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / img_h
    bboxes = bboxes.tolist()

    return bboxes

def convert_bboxLabelme2bboxArr(
    bboxLabelme: Union[List[BBoxLabelmeType], BBoxLabelmeType]
) -> BBoxesArrType:
    bboxes = np.asarray(bboxLabelme).reshape(-1, 4)
    return bboxes

def convert_bboxCoco2bboxArr(
    bboxCoco: Union[List[BBoxCocoType], BBoxCocoType]
) -> BBoxesArrType:
    bboxes = np.asarray(bboxCoco)
    bboxes[:, [2, 3]] = bboxes[:, [0, 1]] + bboxes[:, [2, 3]]
    return bboxes

def convert_bboxYolo2bboxArr(
    bboxYolo: Union[List[BBoxYoloType], BBoxYoloType],
    img_hw: Tuple[int, int]
) -> BBoxesArrType:
    img_h, img_w = img_hw
    bboxes = np.asarray(bboxYolo)
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * img_w
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * img_h
    bboxes[:, [0, 1]] = bboxes[:, [0, 1]] - 0.5 * bboxes[:, [2, 3]]
    bboxes[:, [2, 3]] = bboxes[:, [0, 1]] + bboxes[:, [2, 3]]
    bboxes = bboxes
    return bboxes

def concat_bboxes(
    bboxes_list: List[Union[BBoxArrType, BBoxesArrType]]
) -> BBoxesArrType:
    for i in range(len(bboxes_list)):
        if len(bboxes_list[i].shape) == 1:
            bboxes_list[i] = bboxes_list[i].reshape(-1, 4)
    
    bboxes = np.concat(bboxes_list, axis = 0)

    return bboxes

def ious_bboxes(
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

def _adaptive_thickness(
    img_hw: Tuple[int, int], 
    base_size: int = 1080, 
    base_thickness: int = 2
) -> int:
    """
    根据图像分辨率自动计算线宽。
    
    Args:
        img_hw: 图像的 shape，格式为 (height, width, ...) 或 (height, width)
        base_size: 基准尺寸（例如 1080p 的短边或对角线）
        base_thickness: 在基准尺寸下使用的线宽（例如 2）
    
    Returns:
        int: 自适应的 thickness（至少为 1）
    """
    h, w = img_hw[:2]
    # 方法1：按对角线比例（更通用）
    diag = np.sqrt(h**2 + w**2)
    base_diag = np.sqrt(2) * base_size  # 假设 base_size 是 1080（短边），则对角线约为 1527
    thickness = max(1, int(base_thickness * diag / base_diag))

    # 方法2（可选）：按最大边比例
    # max_side = max(h, w)
    # thickness = max(1, int(base_thickness * max_side / base_size))

    return thickness

def draw_bboxes(
    img: NDArray[np.uint8],
    bboxes: Union[BBoxArrType, BBoxesArrType],
    colors: Union[Literal["default"], Tuple[int, int, int]],
    thickness: Union[Literal["default"], int]
) -> NDArray[np.uint8]:
    BBOX_DEFAULT_COLORS = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (224, 132, 200),
        (73, 242, 180),
        (242, 138, 73),
        (66, 200, 237),
        (237, 66, 92)
    ]

    if isinstance(colors, str):
        if colors == "default":
            colors = BBOX_DEFAULT_COLORS * (len(bboxes) // len(BBOX_DEFAULT_COLORS) + 1)
            colors = colors[:len(bboxes)]
        else:
            raise ValueError("colors")
    else:
        colors = [colors] * len(bboxes)
    
    if isinstance(thickness, str):
        if thickness == "default":
            thickness = _adaptive_thickness(img.shape[:2])
        else:
            raise ValueError("thickness")

    for bbox, color in zip(bboxes, colors):
        x1, y1, x2, y2 = bbox
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    return img