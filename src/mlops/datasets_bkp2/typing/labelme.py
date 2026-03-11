from typing import TypedDict, Union, Optional, Literal, Any, List, Dict, TypeAlias

from mlops.shapes.typing.bboxes import BBoxLabelmeType
from mlops.shapes.typing.polys import PolyLabelmeType


__all__ = [
    "LabelmeShapeType",
    "LabelmeFileType",
    "LabelmeShapeGroupsType"
]


class LabelmeShapeType(TypedDict):
    """
    `LabelmeShapeType`, `dict`
        `points`: `Union[PolyLabelmeType, BBoxLabelmeType]`
        `label`: `str`
        `shape_type`: `Literal["polygon", "rectangle"]`
        `group_id`: `Optional[str]`
        `flags`: `Dict[Any, Any]`
    """
    points: Union[PolyLabelmeType, BBoxLabelmeType]
    label: str
    shape_type: Literal["polygon", "rectangle"]
    group_id: Optional[str]
    flags: Dict[Any, Any]

class LabelmeFileType(TypedDict):
    """
    `LabelmeFileType`, `dict`
        `version`: `str`
        `flags`: `Dict[str, bool]`
        `shapes`: `List[LabelmeShapeType]`
        `imagePath`: `str`
        `imageData`: `Optional[str]`
        `imageHeight`: int
        `imageWidth`: `int`
    """
    version: str
    flags: Dict[str, bool]
    shapes: List[LabelmeShapeType]
    imagePath: str
    imageData: Optional[str]
    imageHeight: int
    imageWidth: int

LabelmeShapeGroupsType: TypeAlias = Dict[Union[str, int], List[LabelmeShapeType]]
"""
`LabelmeShapeGroupsType`
    `Dict[Union[str, int], List[LabelmeShapeType]]`, 
    `{group_id: labelme_shape_dict_list}`,
    `(num_groups, )`
"""