from typing import List, TypedDict, Tuple, TypeAlias


__all__ = [
    "RleDictType",
    "RlesDictType"
]



class RleDictType(TypedDict):
    """
    `RLEType`, `dict`
        `size`: `Tuple[int, int]`, `[img_h, img_w]`
        `counts`: `str`
    """
    size: Tuple[int, int]
    counts: str

RlesDictType: TypeAlias = List[RleDictType]
"""
`RLEsType`
    `List[RleDictType]`, `[rle1, rle2, ...]`
"""