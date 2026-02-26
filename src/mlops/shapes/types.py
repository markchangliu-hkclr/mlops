"""
# Shape Types Module

This module defines dimensions, dtypes, and formats for different kinds of shapes. 

The module is mainly used for type-hinting and regulating APIs for other modules.

## Types 

### Contour
- `ContourType`
- `ContourGroupType`
- `HierarchiesType`

### BBox
- `BBoxArrType`
- `BBoxLabelmeType`
- `BBoxYoloType`
- `BBoxCocoType`
- `BBoxesArrType`

### Masks
- `MaskArrType`
- `MaskImgType`
- `MasksArrType`

### Polys
- `PolyArrType`
- `PolyLabelmeType`
- `PolyCocoType`
- `PolyLabelmeType`

### Rles
- `RleType`
"""

from typing import TypeAlias, TypedDict, List, Tuple

import numpy as np
from numpy.typing import NDArray


ContourType: TypeAlias = NDArray[np.int32]
"""
`NDArray[np.int32], (num_points, 1, 2), each (1, 2) is (x, y)`
"""

class ContourGroupType(TypedDict):
    """
    `{"parent": parent_cnt, "children": child_cnt_list}`
    """
    parent: ContourType
    children: List[ContourType]

HierarchiesType: TypeAlias = NDArray[np.int32]
"""
`NDArray[np.int32], (1, num_cnts, 4)`; 

`hierarchies[0][cnt_i][0]`: 
index of next poly in the same hierarchy, -1 means none. 

`hierarchies[0][cnt_i][1]`: 
index of previous poly in the same hierarchy, -1 means none. 

`hierarchies[0][cnt_i][2]`: 
index of firsh_child poly in the higher level hierarchy, -1 means none. 

`hierarchies[0][cnt_i][3]`: 
index of parent poly in the higher level hierarchy, -1 means none.
"""

BBoxArrType: TypeAlias = NDArray[np.number]
"""
`NDArray[np.number], (4, ), [x1, y1, x2, y2]`
"""

BBoxLabelmeType: TypeAlias = Tuple[Tuple[float, float], Tuple[float, float]]
"""
`Tuple[Tuple[float, float], Tuple[float, float]], (2, (2, )), [[x1, y1], [x2, y2]]`
"""

BBoxCocoType: TypeAlias = Tuple[float, float, float, float]
"""
`Tuple[float, float, float, float], (4, ), [x1, y1, w, h]`
"""

BBoxYoloType: TypeAlias = Tuple[float, float, float, float]
"""
`Tuple[float, float, float, float], (4, ), [x_ctr_norm, y_ctr_norm, w_norm, y_norm]`
"""

BBoxesArrType: TypeAlias = NDArray[np.number]
"""
`NDArray[np.number], (num_bboxes, 4), [[x1, y1, x2, y2], ...]`
"""

MaskArrType: TypeAlias = NDArray[np.bool_]
"""
`NDArray[np.bool_], (img_h, img_w)`
"""

MaskImgType: TypeAlias = NDArray[np.uint8]
"""
`NDArray[np.uint8], (img_h, img_w), 0 ~ 255
"""

MasksArrType: TypeAlias = NDArray[np.bool_]
"""
`NDArray[np.bool_], (num_masks, img_h, img_w)`
"""

PolyArrType: TypeAlias = NDArray[np.integer]
"""
`NDArray[np.integer], (num_points, 2), [[x, y], ...]`
"""

PolyLabelmeType: TypeAlias = List[Tuple[float, float]]
"""
`List[Tuple[float, float]]`, `(num_points, (2, ))`, `[[x1, y1], [x2, y2], ...]`
"""

PolyCocoType: TypeAlias = List[float]
"""
`List[float], (num_points * 2, ), [x1, y1, x2, y2, x3, y3, ...]`
"""

PolyYoloType: TypeAlias = List[float]
"""
`List[float], (num_points * 2, ), [x1_norm, y1_norm, x2_norm, y2_norm, ...]`
"""

class RleType(TypedDict):
    """
    `"size": Tuple[int, int], [img_h, img_w]`
    `"counts": str`
    """
    size: Tuple[int, int]
    counts: str