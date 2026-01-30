from typing import TypedDict, Literal, Union, List, TypeAlias

from mlops.shapes.typing.bboxes import BBoxCocoType
from mlops.shapes.typing.polys import PolyCocoType


__all__ = [
    "COCO_IMG_TEMPLATE",
    "COCO_CAT_TEMPLATE",
    "COCO_ANN_TEMPLATE",
    "COCO_TEMPLATE",
    "CocoImgType",
    "CocoCatType",
    "CocoAnnType",
    "CocoFileType"
]


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


class CocoImgType(TypedDict):
    """
    `CocoImgType`, `dict`
        `height`: `int`
        `width`: `int`
        `id`: `int`
        `file_name`: `str`
    """
    height: int
    width: int
    id: int
    file_name: str

class CocoCatType(TypedDict):
    """
    `CocoCatType`, `dict`
        `id`: `int`
        `name`: `str`
    """
    id: int
    name: str

class CocoAnnType(TypedDict):
    """
    `CocoAnnType`, `dict`
        `id`: `int`
        `iscrowd`: `Literal[0, 1]`
        `image_id`: `int`
        `category_id`: `int`
        `area`: `int`,
        `bbox`: `BBoxCocoType`,
        `segmentation`: `Union[PolysCocoType, EmptyListType]`
    """
    id: int
    iscrowd: Literal[0, 1]
    image_id: int
    category_id: int
    area: int
    bbox: BBoxCocoType
    segmentation: Union[List[PolyCocoType], EmptyListType]

class CocoFileType(TypedDict):
    """
    `CocoFileType`, `dict`
        `images`: `List[CocoImgDict]`
        `categories`: `List[CocoCatDict]`
        `annotations`: `List[CocoAnnDict]`
    """
    images: List[CocoImgType]
    categories: List[CocoCatType]
    annotations: List[CocoAnnType]