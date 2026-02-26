from typing import List, TypedDict, Tuple, TypeAlias


__all__ = [
    "RleType",
    "RlesType"
]



class RleType(TypedDict):
    """
    `RLEType`, `dict`
        `size`: `Tuple[int, int]`, `[img_h, img_w]`
        `counts`: `str`
    """
    size: Tuple[int, int]
    counts: str

RlesType: TypeAlias = List[RleType]
"""
`RLEsType`
    `List[RleDictType]`, `[rle1, rle2, ...]`
"""