from typing import List, Tuple, TypeAlias


__all__ = [
    "YoloBBoxType",
    "YoloPolyType",
]


YoloBBoxType: TypeAlias = Tuple[int, float, float, float, float]
"""
`YoloBBoxType`
    `Tuple[int, float, float, float, float]`, `(5, )`, 
    `[cat_id, x_ctr_norm, y_ctr_norm, w_norm, y_norm]`
"""

YoloPolyType: TypeAlias = List[float]
"""
`YoloPolyType`
    `Tuple[int, float, ...]`, `(1 + num_points * 2, )`, 
    `[cat_id, x1_norm, y1_norm, x2_norm, y2_norm, ...]`
"""
