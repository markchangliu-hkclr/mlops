from typing import Literal, Union

import numpy as np
from numpy.typing import NDArray

from mlops.shapes.typing.bboxes import BBoxesArrXYXYType
from mlops.shapes.typing.masks import MasksArrType


__all__ = [
    "get_ious",
    "get_ious_bboxes",
    "get_ious_masks"
]


def get_ious(
    shapesA: Union[BBoxesArrXYXYType, MasksArrType],
    shapesB: Union[BBoxesArrXYXYType, MasksArrType],
    shape_format: Literal["bbox", "poly"],
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
    assert shape_format in ["bbox", "poly"]
    assert mode in ["iou", "iof"]

    if shape_format == "bbox":
        ious = get_ious_bboxes(shapesA, shapesB, mode)
    else:
        ious = get_ious_masks(shapesA, shapesB, mode)
    
    return ious

def get_ious_bboxes(
    bboxesA: BBoxesArrXYXYType,
    bboxesB: BBoxesArrXYXYType,
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

    # Expand dimensions to enable broadcasting
    # bboxesA: (num_bboxesA, 1, 4)
    # bboxesB: (1, num_bboxesB, 4)
    A = bboxesA[:, None, :]  # shape (num_bboxesA, 1, 4)
    B = bboxesB[None, :, :]  # shape (1, num_bboxesB, 4)
    
    # Compute intersection coordinates
    # (num_bboxesA, num_bboxesB)
    x1_inter = np.maximum(A[..., 0], B[..., 0])
    y1_inter = np.maximum(A[..., 1], B[..., 1])
    x2_inter = np.minimum(A[..., 2], B[..., 2])
    y2_inter = np.minimum(A[..., 3], B[..., 3])
    
    # Compute intersection area
    w_inter = np.maximum(0, x2_inter - x1_inter)
    h_inter = np.maximum(0, y2_inter - y1_inter)
    area_inter = w_inter * h_inter

    if mode == "iof":
        areasA = np.prod(bboxesA[:, [2, 3]] - bboxesA[:, [0, 1]], axis = 1)
        iofs = area_inter / (areasA[:, None] + 1e-8)
        return iofs
    elif mode == "iou":
        # Compute union area
        x1_union = np.minimum(A[..., 0], B[..., 0])
        y1_union = np.minimum(A[..., 1], B[..., 1])
        x2_union = np.maximum(A[..., 2], B[..., 2])
        y2_union = np.maximum(A[..., 3], B[..., 3])

        w_union = np.maximum(0, x2_union - x1_union)
        h_union = np.maximum(0, y2_union - y1_union)
        area_union = w_union * h_union

        ious = area_inter / (area_union + 1e-8)
        return ious
    else:
        raise NotImplementedError

def get_ious_masks(
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
    elif mode == "iof":
        iofs = inter_area / (masksA_area + 1e-8)
        return iofs
    else:
        raise NotImplementedError
