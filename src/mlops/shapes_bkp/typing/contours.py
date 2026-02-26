from typing import List, Literal, Tuple, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt


__all__ = [
    "ContourType",
    "ContourGroupType",

    "ContoursType",
    "ContourGroupsType",
    
    "HierarchiesType",
    "ContourShapeType",
    "HierarchiesShapeType",
]


ContourType: TypeAlias = npt.NDArray[np.int32]
"""
`ContourType`
    `NDArray[np.int32]`, `(num_points, 1, 2)`, each `(1, 2)` is `(x, y)`
"""

class ContourGroupType(TypedDict):
    """
    `ContourGroupType`
        `List[Dict[str, Union[ContourType, ContoursType]]]`, 
        `{"parent": parent_cnt, "children": child_cnt_list}`
    """
    parent: ContourType
    children: List[ContourType]

ContoursType: TypeAlias = List[npt.NDArray[np.int32]]
"""
`ContoursType`
    `List[NDArray[np.int32]]`, `(num_cnts, (num_points (1, 2)))`
"""

ContourGroupsType: TypeAlias = List[ContourGroupType]
"""
`ContourGroupsType`
    `List[ContourGroupType]`, `(num_groups, )`,
    `[{"parent": parent_cnt, "children": child_cnt_list}, ...]`
"""

HierarchiesType: TypeAlias = npt.NDArray[np.int32]
"""
`HierarchiesType`
    `NDArray[np.int32]`, `(1, num_cnts, 4)`; \n
    `hierarchies[0][cnt_i][0]`: index of next poly in the same hierarchy, `-1` means none. \n
    `hierarchies[0][cnt_i][1]`: index of previous poly in the same hierarchy, `-1` means none. \n
    `hierarchies[0][cnt_i][2]`: index of firsh_child poly in the higher level hierarchy, `-1` means none. \n
    `hierarchies[0][cnt_i][3]`: index of parent poly in the higher level hierarchy, `-1` means none. \n
"""

ContourShapeType: TypeAlias = Tuple[int, Literal[1], Literal[2]]
"""
`ContourShapeType`
    `Tuple[int, Literal[1], Literal[2]]`, `[num_points, 1, 2]`
"""

HierarchiesShapeType: TypeAlias = Tuple[Literal[1], int, Literal[4]]
"""
`HierarchiesShapeType`
    `Tuple[Literal[1], int, Literal[4]]`, [1, num_cnts, 4]
"""