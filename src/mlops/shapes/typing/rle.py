from typing import List, TypedDict, Tuple

try:
    from typing import TypeAlias
except:
    from typing_extensions import TypeAlias


__all__ = [
    "RLEType",
    "RLEsType"
]



class RLEType(TypedDict):
    """
    `RLEType`, `dict`
        `size`: `Tuple[int, int]`, `[img_h, img_w]`
        `counts`: `str`
    """
    size: Tuple[int, int]
    counts: str

RLEsType: TypeAlias = List[RLEType]
"""
`RLEsType`
    `List[Dict[str, Any]]`, `[rle1, rle2, ...]`
"""