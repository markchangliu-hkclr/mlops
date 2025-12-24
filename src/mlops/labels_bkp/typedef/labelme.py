from typing import TypedDict, Union, Optional, Literal, Any, List, Dict

try:
    from typing import TypeAlias
except:
    from typing_extensions import TypeAlias

from mlops.shapes.typedef.bboxes import BBoxLabelmeType
from mlops.shapes.typedef.polys import PolyLabelmeType
from mlops.shapes.typedef.rles import RLEType
from mlops.shapes.typedef.contours import ContourType


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