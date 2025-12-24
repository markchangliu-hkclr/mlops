from typing import TypedDict, Union, Optional, Literal, Any, List, Dict

try:
    from typing import TypeAlias
except:
    from typing_extensions import TypeAlias

from mlops.shapes.typing import BBoxLabelmeType, PolyLabelmeType


__all__ = [
    "LabelmeShapeDictType",
    "LabelmeDictType",
    "LabelmeShapeGroupsType"
]


class LabelmeShapeDictType(TypedDict):
    """
    `LabelmeShapeDictType`, `dict`
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

class LabelmeDictType(TypedDict):
    """
    `LabelmeDictType`, `dict`
        `version`: `str`
        `flags`: `Dict[str, bool]`
        `shapes`: `List[LabelmeShapeDictType]`
        `imagePath`: `str`
        `imageData`: `Optional[str]`
        `imageHeight`: int
        `imageWidth`: `int`
    """
    version: str
    flags: Dict[str, bool]
    shapes: List[LabelmeShapeDictType]
    imagePath: str
    imageData: Optional[str]
    imageHeight: int
    imageWidth: int

LabelmeShapeGroupsType: TypeAlias = Dict[Union[str, int], List[LabelmeShapeDictType]]
"""
`LabelmeShapeGroupsType`
    `Dict[Union[str, int], List[LabelmeShapeDictType]]`, 
    `{group_id: labelme_shape_dict_list}`,
    `(num_groups, )`
"""