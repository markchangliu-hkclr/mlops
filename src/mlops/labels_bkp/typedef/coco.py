from typing import TypedDict, Literal, Union, List

try:
    from typing import TypeAlias
except:
    from typing_extensions import TypeAlias

from mlops.shapes.typedef.bboxes import BBoxCocoType
from mlops.shapes.typedef.polys import PolysCocoType
from mlops.shapes.typedef.rles import RLEsType, RLEType


COCO_IMG_TEMPLATE = {
    "height": 0,
    "width": 0,
    "id": 0,
    "file_name": ""
}

COCO_CAT_TEMPLATE = {
    "id": 0,
    "name": ""
}

COCO_ANN_TEMPLATE = {
    "id": 0,
    "iscrowd": 0,
    "image_id": 0,
    "area": 0,
    "bbox": [],
    "segmentation": [],
}

COCO_TEMPLATE = {
    "images": COCO_IMG_TEMPLATE,
    "categories": COCO_CAT_TEMPLATE,
    "annotations": COCO_ANN_TEMPLATE
}


EmptyListType: TypeAlias = List[int]


class CocoImgDictType(TypedDict):
    """
    `CocoImgDictType`, `dict`
        `height`: `int`
        `width`: `int`
        `id`: `int`
        `file_name`: `str`
    """
    height: int
    width: int
    id: int
    file_name: str

class CocoCatDictType(TypedDict):
    """
    `CocoCatDictType`, `dict`
        `id`: `int`
        `name`: `str`
    """
    id: int
    name: str

class CocoAnnDictType(TypedDict):
    """
    `CocoAnnDictType`, `dict`
        `id`: `int`
        `iscrowd`: `Literal[0, 1]`
        `image_id`: `int`
        `category_id`: `int`
        `area`: `int`,
        `bbox`: `BBoxCocoType`,
        `segmentation`: `Union[PolysCocoType, RLEsType, EmptyListType]`
    """
    id: int
    iscrowd: Literal[0, 1]
    image_id: int
    category_id: int
    area: int
    bbox: BBoxCocoType
    segmentation: Union[PolysCocoType, RLEsType, RLEType, EmptyListType]

class CocoDictType(TypedDict):
    """
    `CocoDictType`, `dict`
        `images`: `List[CocoImgDict]`
        `categories`: `List[CocoCatDict]`
        `annotations`: `List[CocoAnnDict]`
    """
    images: List[CocoImgDictType]
    categories: List[CocoCatDictType]
    annotations: List[CocoAnnDictType]