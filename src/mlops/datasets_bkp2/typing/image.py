from typing import TypedDict, Tuple


class ImgMetaType(TypedDict):
    """
    - `img_hw: Tuple[int, int]`
    - `img_p: str`
    """
    img_hw: Tuple[int, int]
    img_p: str