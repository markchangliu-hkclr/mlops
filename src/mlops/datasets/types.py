from typing import TypeAlias, TypedDict, List, Tuple, Union, Literal, Dict, Any

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
`Tuple[float, float, float, float], (4, ), [x_ctr_norm, y_ctr_norm, w_norm, h_norm]`
"""

BBoxType: TypeAlias = Union[BBoxArrType, BBoxCocoType, BBoxCocoType, BBoxLabelmeType]
"""
`Union[BBoxArrType, BBoxCocoType, BBoxCocoType, BBoxLabelmeType]`
"""

BBoxesArrType: TypeAlias = NDArray[np.number]
"""
`NDArray[np.number], (num_bboxes, 4), [[x1, y1, x2, y2], ...]`
"""

BBoxesType: TypeAlias = Union[BBoxesArrType, List[BBoxLabelmeType], List[BBoxCocoType], List[BBoxCocoType], List[BBoxYoloType]]
"""
`Union[BBoxesArrType, List[BBoxLabelmeType], List[BBoxCocoType], List[BBoxCocoType], List[BBoxYoloType]]`
"""

BBoxFormat: TypeAlias = Literal["arr", "labelme", "coco", "yolo"]
"""
`Literal["arr", "labelme", "coco", "yolo"]`
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

MaskType: TypeAlias = Union[MaskArrType, "RleType", PolyArrType, PolyLabelmeType, PolyCocoType, PolyYoloType]
"""
`Union[MaskArrType, "RleType", PolyArrType, PolyLabelmeType, PolyCocoType, PolyYoloType]`
"""

MasksType: TypeAlias = 

PolyFormat: TypeAlias = Literal["arr", "labelme", "coco", "yolo"]
"""
`Literal["arr", "labelme", "coco", "yolo"]`
"""

class RleType(TypedDict):
    """
    `"size": Tuple[int, int], [img_h, img_w]`
    `"counts": str`
    """
    size: Tuple[int, int]
    counts: str

class LabelmeFileType(TypedDict):
    """
    `version: str`
    `flags: Dict[str, bool]`
    `shapes: List[LabelmeShapeDictType]`
    `imagePath: str`
    `imageData: Optional[str]`
    `imageHeight: int`
    `imageWidth: int`
    """
    version: str
    flags: Dict[str, bool]
    shapes: List["LabelmeShapeType"]
    imagePath: str
    imageData: Union[str, None]
    imageHeight: int
    imageWidth: int

class LabelmeShapeType(TypedDict):
    """
    `points: Union[PolyLabelmeType, BBoxLabelmeType]`
    `label: str`
    `shape_type: Literal["polygon", "rectangle"]`
    `group_id: Optional[str]`
    `flags: Dict[Any, Any]`
    """
    points: Union[PolyLabelmeType, BBoxLabelmeType]
    label: str
    shape_type: Literal["polygon", "rectangle"]
    group_id: Union[str, None]
    flags: Dict[Any, Any]

class LabelmeShapeGroupType(TypedDict):
    """
    `group_id: int`
    `shapes: List[LabelmeShapeType]`
    """
    group_id: int
    shapes: List[LabelmeShapeType]

class CocoImgType(TypedDict):
    """
    `height: int`
    `width: int`
    `id: int`
    `file_name: str`
    """
    height: int
    width: int
    id: int
    file_name: str

class CocoCatType(TypedDict):
    """
    `id: int`
    `name: str`
    """
    id: int
    name: str

class CocoAnnType(TypedDict):
    """
    `id: int`
    `iscrowd: Literal[0, 1]`
    `image_id: int`
    `category_id: int`
    `area: int`,
    `bbox: BBoxCocoType`,
    `segmentation: PolysCocoType`
    """
    id: int
    iscrowd: Literal[0, 1]
    image_id: int
    category_id: int
    area: int
    bbox: BBoxCocoType
    segmentation: List[PolyCocoType]

class CocoFileType(TypedDict):
    """
    `images: List[CocoImgDict]`
    `categories: List[CocoCatDict]`
    `annotations: List[CocoAnnDict]`
    """
    images: List[CocoImgType]
    categories: List[CocoCatType]
    annotations: List[CocoAnnType]

YoloDetAnnType: TypeAlias = Tuple[int, float, float, float, float]
"""
`Tuple[int, float, float, float, float], (5, ), `
`[cat_id, x_ctr_norm, y_ctr_norm, w_norm, h_norm]`
"""

YoloSegAnnType: TypeAlias = List[float]
"""
`Tuple[int, float, ...], (1 + num_points * 2, ),`
`[cat_id, x1_norm, y1_norm, x2_norm, y2_norm, ...]`
"""