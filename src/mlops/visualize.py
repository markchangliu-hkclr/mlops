from typing import Tuple, List, Union, Optional, Literal

import cv2
import numpy as np


TXT_DEFAULT_SIZE = 0.5
TXT_DEFAULT_COLOR = (255, 255, 255)
TXT_DEFAULT_THICKNESS = 2
BBOX_DEFAULT_THICKNESS = 2
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


def draw_scores(
    img: np.ndarray,
    scores: np.ndarray,
    cat_names: np.ndarray,
    txt_xys: np.ndarray,
    txt_color: Union[Tuple[int, int, int], Literal["bbox", "default"]] = "default",
    txt_thickness: Union[int, Literal["default"]] = "default"
) -> np.ndarray:
    """
    Args:
        `img`: `np.ndarray`, `(img_h, img_w, 3)`
        `scores`: `np.ndarray`, `(num_bboxes,)`
        `cat_names`: `np.ndarray`, `(num_bboxes,)`
        `txt_xys`: `np.ndarray`, `(num_bboxes, 2)`, `(x1, y1)`
        `txt_color`: `Union[Tuple[int, int, int], Literal["bbox", "default"]]`, 
        BGR color code
        `txt_thickness`: `Union[int, Literal["default"]]`

    Returns:
        `img_with_scores`: `np.ndarray`, `(img_h, img_w, 3)`
    """
    scores = np.round(scores, 3)

    if isinstance(txt_color, str):
        if txt_color == "bbox":
            txt_colors = BBOX_DEFAULT_COLORS * (len(scores) // len(BBOX_DEFAULT_COLORS) + 1)
            txt_colors = txt_colors[:len(scores)]
        elif txt_color == "default":
            txt_colors = [TXT_DEFAULT_COLOR] * len(scores)
        else:
            raise NotImplementedError
    elif isinstance(txt_color, tuple):
        txt_colors = [txt_color] * len(scores)
    else:
        raise NotImplementedError
    
    if isinstance(txt_thickness, str) and txt_thickness == "default":
        txt_thickness = TXT_DEFAULT_THICKNESS

    for score, cat_name, txt_xy, txt_color in zip(scores, cat_names, txt_xys, txt_colors):
        score = f"{cat_name}: {str(score)}"
        img = cv2.putText(
            img, score, txt_xy, cv2.FONT_HERSHEY_SIMPLEX,
            TXT_DEFAULT_SIZE, txt_color, txt_thickness, cv2.LINE_AA
        )
    
    return img


def draw_bboxes(
    img: np.ndarray,
    bboxes: np.ndarray,
    bbox_color: Union[Tuple[int, int, int], Literal["default"]] = "default",
    bbox_thickness: Union[int, Literal["default"]] = "default"
) -> np.ndarray:
    """
    Args
    -----
    - `img`: `np.ndarray`, `(img_h, img_w, 3)`
    - `bboxes`: `np.ndarray`, `(num_bboxes, 4)`, x1y1x2y2
    - `bbox_color`: `Union[Tuple[int, int, int], Literal["default"]]`, BGR color code
    - `bbox_thickness`: `Union[int, Literal["default"]]`

    Returns
    -------
    - `img_with_bboxes`: `np.ndarray`, `(img_h, img_w, 3)`
    """
    if isinstance(bbox_color, str):
        if bbox_color == "default":
            bbox_colors = BBOX_DEFAULT_COLORS * (len(bboxes) // len(BBOX_DEFAULT_COLORS) + 1)
            bbox_colors = bbox_colors[:len(bboxes)]
    else:
        bbox_colors = [bbox_color] * len(bboxes)
    
    if isinstance(bbox_thickness, str) and bbox_thickness == "default":
        bbox_thickness = BBOX_DEFAULT_THICKNESS

    for bbox, bbox_color in zip(bboxes, bbox_colors):
        x1, y1, x2, y2 = bbox
        img = cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, bbox_thickness)

    return img

def draw_polys(
    img: np.ndarray,
    polys: List[np.ndarray],
    poly_color: Optional[Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    Args
    -----
    - `img`: `np.ndarray`, `(img_h, img_w, 3)`
    - `polys`: `List[np.ndarray]`, `(num_objs, (num_points, 2))`, 
    - `poly_color`: `Tuple[int, int, int]`, BGR color code

    Returns
    -------
    - `img_with_polys`: `np.ndarray`, `(img_h, img_w, 3)`
    """
    thickness = 2

    if poly_color is None:
        poly_colors = BBOX_DEFAULT_COLORS * (len(polys) // len(BBOX_DEFAULT_COLORS) + 1)
        poly_colors = poly_colors[:len(polys)]
    else:
        poly_colors = [poly_color] * len(polys)

    for poly, poly_color in zip(polys, poly_colors):
        poly = poly.reshape((-1, 1, 2))
        img = cv2.polylines(img, [poly], True, poly_color, thickness)

    return img

def draw_masks(
    img: np.ndarray,
    masks: np.ndarray,
    mask_color: np.ndarray = None
) -> np.ndarray:
    """
    Args
    -----
    - `img`: `np.ndarray`, `(img_h, img_w, 3)`
    - `masks`: `np.ndarray`, `(num_masks, img_h, img_w)`
    - `mask_color`: `Tuple[int, int, int]`, BGR color code

    Returns
    -------
    - `img_with_masks`: `np.ndarray`, `(img_h, img_w, 3)`
    """
    if isinstance(mask_color, str) and mask_color == "default":
        mask_colors = BBOX_DEFAULT_COLORS * (len(masks) // len(BBOX_DEFAULT_COLORS) + 1)
        mask_colors = mask_colors[:len(masks)]
    else:
        mask_colors = [mask_color] * len(masks)

    for mask, mask_color in zip(masks, mask_colors):
        mask_color = np.array(mask_color, dtype='uint8')
        masked_img = np.where(mask[..., None], mask_color, img)
        img = cv2.addWeighted(img, 0.7, masked_img, 0.3, 0)

    return img

def draw_insts(
    img: np.ndarray,
    scores: np.ndarray,
    cat_names: np.ndarray,
    bboxes: np.ndarray,
    masks: Optional[np.ndarray],
    bbox_color: Union[Tuple[int, int, int], Literal["default"]],
    mask_color: Union[Tuple[int, int, int], Literal["default"]]
) -> np.ndarray:
    if len(scores) == 0:
        return img
    
    img = np.copy(img)

    img = draw_scores(img, scores, cat_names, bboxes[:, :2])
    img = draw_bboxes(img, bboxes, bbox_color)

    if masks is not None:
        img = draw_masks(img, masks, mask_color)
        
    return img