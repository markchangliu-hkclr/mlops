from typing import List, Tuple

try:
    from typing import TypeAlias
except:
    from typing_extensions import TypeAlias


__all__ = [
    "PolyLabelmeType",
    "PolyCocoType",
    "PolyYoloType",

    "PolysLabelmeType",
    "PolysCocoType",
    "PolysYoloType",
]



PolyLabelmeType: TypeAlias = List[Tuple[float, float]]
"""
`PolyLabelmeType`
    `List[Tuple[float, float]]`, `(num_points, (2, ))`, `[[x1, y1], [x2, y2], ...]`
"""

PolysLabelmeType: TypeAlias = List[List[Tuple[float, float]]]
"""
`PolysLabelmeType`
    `List[List[Tuple[float, float]]]`, `(num_polys, (num_points, (2, )))`
"""

PolyCocoType: TypeAlias = List[float]
"""
`PolyCocoType`
    `List[float]`, `(num_points * 2, )`, `[x1, y1, x2, y2, ...]`
"""

PolysCocoType: TypeAlias = List[List[float]]
"""
`PolysCocoType`
    `List[List[float]]`, `(num_polys, (num_points * 2, ))`
"""

PolyYoloType: TypeAlias = List[float]
"""
`PolysYoloType`
    `List[float]`, `(num_points * 2, )`, `[x1_norm, y1_norm, x2_norm, y2_norm, ...]`
"""

PolysYoloType: TypeAlias = List[List[float]]
"""
`PolysYoloType`
    `List[List[float]]`, `(num_polys, (num_points * 2, ))`
"""