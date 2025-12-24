from typing import Union, List, Tuple
try:
    from typing import TypeAlias
except:
    from typing_extensions import TypeAlias


YoloBBoxLabelType: TypeAlias = Tuple[int, float, float, float, float]
"""
`YoloBBoxLabelType`
    `Tuple[int, float, float, float, float]`, `(5, )`, 
    `[cat_id, x_ctr_norm, y_ctr_norm, w_norm, y_norm]`
"""

YoloBBoxLabelsType: TypeAlias = List[YoloBBoxLabelType]
"""
`YoloBBoxLabelType`
    `List[YoloBBoxLabelType]`, `(num_labels, (5, ))`, 
    `[[cat_id, x_ctr_norm, y_ctr_norm, w_norm, y_norm], ...]`
"""

YoloPolyLabelType: TypeAlias = List[float]
"""
`YoloPolyLabelType`
    `Tuple[int, float, ...]`, `(1 + num_points * 2, )`, 
    `[cat_id, x1_norm, y1_norm, x2_norm, y2_norm, ...]`
"""

YoloPolyLabelsType: TypeAlias = List[YoloPolyLabelType]
"""
`YoloPolyLabelsType`
    `List[YoloPolyLabelType]`, `(num_labels, (1 + num_points * 2, ))`, 
    `[[cat_id, x1_norm, y1_norm, x2_norm, y2_norm, ...], ...]`
"""