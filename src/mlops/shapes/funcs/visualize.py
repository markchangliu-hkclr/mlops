from typing import Tuple, List, Union, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from mlops.shapes.typing.others import *
from mlops.shapes.typing.bboxes import BBoxesArrXYXYType
from mlops.shapes.typing.masks import MasksArrType
from mlops.shapes.typing.polys import PolyArrType
from mlops.shapes.structs.instances import Instances


DEFAULT_TXT_SIZE = 0.5
DEFAULT_TXT_COLOR = (255, 255, 255)
DEFAULT_TXT_THICKNESS = 2
DEFAULT_THICKNESS = 2
DEFAULT_COLORS = [
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


def adaptive_thickness(
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

def adaptive_font_scale(
    img_hw: Tuple[int, int], 
    base_size: int = 1080, 
    base_font_scale: float = 1.0
) -> float:
    """
    根据图像分辨率自动计算 OpenCV putText 的 fontScale。
    
    Args:
        img_hw: 图像 shape，格式为 (height, width, ...) 或 (height, width)
        base_size: 基准尺寸（通常取短边或对角线，这里默认按短边）
        base_font_scale: 在基准尺寸图像上看起来合适的 fontScale（例如 1.0 对应 1080p 短边）
    
    Returns:
        float: 自适应的 fontScale（最小不低于 0.3，避免太小看不清）
    """
    h, w = img_hw[:2]
    short_side = min(h, w)
    
    # 按短边比例缩放（更符合文字显示习惯）
    scale = short_side / base_size
    font_scale = base_font_scale * scale
    
    # 设置下限，避免在极小图上文字完全看不见
    return max(0.3, font_scale)

def draw_confs_categories(
    bgr: NDArray[np.uint8],
    confs: ConfidencesArrType,
    cat_ids: CategoryIDsArrType,
    xys: NDArray[np.integer],
    colors: Union[Tuple[int, int, int], Literal["default"]],
    thickness: Union[int, Literal["default"]],
    font_scale: Union[float, Literal["default"]]
) -> NDArray[np.uint8]:
    """
    Args
    -----
    - `bgr`: `np.ndarray`, `(img_h, img_w, 3)`
    - `scores`: `np.ndarray`, `(num_bboxes,)`
    - `cat_ids`: `np.ndarray`, `(num_bboxes,)`
    - `xys`: `np.ndarray`, `(num_bboxes, 2)`, `(x1, y1)`
    - `colors`: `Union[Tuple[int, int, int], Literal["default"]]`, BGR color code
    - `thickness`: `Union[int, Literal["default"]]`
    - `font_scale`: `Union[float, Literal["default"]]`

    Returns
    -----
    - `img_with_scores`: `np.ndarray`, `(img_h, img_w, 3)`
    """
    if len(confs) == 0:
        return bgr

    confs = np.round(confs, 3)

    if isinstance(colors, str):
        if colors == "default":
            colors = DEFAULT_COLORS * (len(confs) // len(DEFAULT_COLORS) + 1)
            colors = colors[:len(confs)]
        else:
            raise ValueError("colors")
    elif isinstance(colors, tuple):
        colors = [colors] * len(confs)
    
    if isinstance(thickness, str) and thickness == "default":
        thickness = adaptive_thickness(bgr.shape[:2])
    elif not isinstance(thickness, int):
        raise ValueError("line_thickness")
    
    if isinstance(font_scale, str) and font_scale == "default":
        font_scale = adaptive_font_scale(bgr.shape[:2])
    elif not isinstance(font_scale, (int, float)):
        raise ValueError("font_scale")

    for conf, cat_id, xy, color in zip(confs, cat_ids, xys, colors):
        score = f"{cat_id}: {str(score)}"
        bgr = cv2.putText(
            bgr, score, xy, cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, color, thickness, cv2.LINE_AA
        )
    
    return bgr

def draw_bboxes(
    bgr: NDArray[np.uint8],
    bboxes: BBoxesArrXYXYType,
    colors: Union[Tuple[int, int, int], Literal["default"]],
    thickness: Union[int, Literal["default"]]
) -> NDArray[np.uint8]:
    """
    Args
    -----
    - `bgr`: `np.ndarray`, `(img_h, img_w, 3)`
    - `bboxes`: `np.ndarray`, `(num_bboxes, 4)`, x1y1x2y2
    - `colors`: `Union[Tuple[int, int, int], Literal["default"]]`, BGR color code
    - `thickness`: `Union[int, Literal["default"]]`

    Returns
    -------
    - `bgr_with_bboxes`: `np.ndarray`, `(img_h, img_w, 3)`
    """
    if len(bboxes) == 0:
        return bgr

    if isinstance(colors, str):
        if colors == "default":
            colors = DEFAULT_COLORS * (len(bboxes) // len(DEFAULT_COLORS) + 1)
            colors = colors[:len(bboxes)]
        else:
            raise ValueError("colors")
    elif isinstance(colors, tuple):
        colors = [colors] * len(bboxes)
    
    if isinstance(thickness, str) and thickness == "default":
        thickness = adaptive_thickness(bgr.shape[:2])
    elif not isinstance(thickness, int):
        raise ValueError("line_thickness")

    for bbox, color in zip(bboxes, colors):
        x1, y1, x2, y2 = bbox
        bgr = cv2.rectangle(bgr, (x1, y1), (x2, y2), color, thickness)

    return bgr

def draw_polys(
    bgr: NDArray[np.uint8],
    polys: List[PolyArrType],
    colors: Union[Tuple[int, int, int], Literal["default"]],
    thickness: Union[int, Literal["default"]]
) -> np.ndarray:
    """
    Args
    -----
    - `bgr`: `np.ndarray`, `(img_h, img_w, 3)`
    - `polys`: `List[PolyArrType]`, `(num_objs, (num_points, 2))`, 
    - `colors`: `Union[Tuple[int, int, int], Literal["default"]]`, BGR color code
    - `thickness`: `Union[int, Literal["default"]]`

    Returns
    -------
    - `bgr_with_polys`: `np.ndarray`, `(img_h, img_w, 3)`
    """
    if len(polys) == 0:
        return bgr

    if isinstance(colors, str):
        if colors == "default":
            colors = DEFAULT_COLORS * (len(polys) // len(DEFAULT_COLORS) + 1)
            colors = colors[:len(polys)]
        else:
            raise ValueError("colors")
    elif isinstance(colors, tuple):
        colors = [colors] * len(polys)
    
    if isinstance(thickness, str) and thickness == "default":
        thickness = adaptive_thickness(bgr.shape[:2])
    elif not isinstance(thickness, int):
        raise ValueError("thickness")

    for poly, color in zip(polys, colors):
        poly = poly.reshape((-1, 1, 2))
        bgr = cv2.polylines(bgr, [poly], True, color, thickness)

    return bgr

def draw_masks(
    bgr: NDArray[np.uint8],
    masks: MasksArrType,
    colors: Union[Tuple[int, int, int], Literal["default"]],
) -> np.ndarray:
    """
    Args
    -----
    - `bgr`: `np.ndarray`, `(img_h, img_w, 3)`
    - `masks`: `np.ndarray`, `(num_masks, img_h, img_w)`
    - `colors`: `Union[Tuple[int, int, int], Literal["default"]]`, BGR color code

    Returns
    -------
    - `img_with_masks`: `np.ndarray`, `(img_h, img_w, 3)`
    """
    if len(masks) == 0:
        return bgr

    if isinstance(colors, str):
        if colors == "default":
            colors = DEFAULT_COLORS * (len(masks) // len(DEFAULT_COLORS) + 1)
            colors = colors[:len(masks)]
        else:
            raise ValueError("colors")
    elif isinstance(colors, tuple):
        colors = [colors] * len(masks)

    for mask, color in zip(masks, colors):
        color = np.array(color, dtype='uint8')
        masked_img = np.where(mask[..., None], color, bgr)
        bgr = cv2.addWeighted(bgr, 0.7, masked_img, 0.3, 0)

    return bgr

def draw_instances(
    bgr: NDArray[np.uint8],
    instances: Instances,
    colors: Union[Tuple[int, int, int], Literal["default"]],
    thickness: Union[int, Literal["default"]],
    font_scale: Union[float, Literal["default"]]
) -> NDArray[np.uint8]:
    if len(instances) == 0:
        return bgr

    bgr = draw_confs_categories(
        bgr, instances.confs, instances.cat_ids, instances.bboxes[:, :2] + 2,
        colors, thickness, font_scale
    )
    bgr = draw_bboxes(bgr, instances.bboxes, colors, thickness)
    
    if instances.masks is not None:
        bgr = draw_masks(bgr, instances.masks, colors)
        
    return bgr