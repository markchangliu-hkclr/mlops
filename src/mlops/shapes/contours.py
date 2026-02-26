"""
# The Contour Module

## Funcs

### Convert
- `convert_contour2mask`
- `convert_contour2poly`
- `convert_contour2rle`

### Merge
- `merge_contours`
"""

import copy
from typing import Tuple, List, Union

import cv2
import numpy as np
import pycocotools.mask as pycocomask

from mlops.shapes.types import ContourType, ContourGroupType, HierarchiesType
from mlops.shapes.types import MasksArrType
from mlops.shapes.types import PolyArrType
from mlops.shapes.types import RleType


def _is_clockwise(contour: ContourType) -> bool:
    contour_next = np.empty_like(contour)
    point_next_idx = list(range(1, len(contour) + 1))
    point_next_idx[-1] = 0
    contour_next[:, ...] = contour_next[point_next_idx, ...]

    shoelace_value = contour_next - contour
    shoelace_value = shoelace_value[:, :, 0] * shoelace_value[:, :, 1]
    shoelace_value = np.sum(shoelace_value).item()

    clockwise_flag = shoelace_value < 0

    return clockwise_flag

def _get_closest_point_idx(
    cnt1: ContourType,
    cnt2: ContourType
) -> Tuple[int, int]:
    # cnt1 shape: (n, 1, 2)
    # cnt2 shape: (1, m, 2)
    # dist_mat shape: (n, m)
    cnt2 = np.transpose(cnt2, (1, 0, 2))

    dist_mat = np.square(cnt1 - cnt2)
    dist_mat = dist_mat[..., 0] + dist_mat[..., 1]

    idx = np.argmin(dist_mat)
    idx1, idx2 = np.unravel_index(idx, dist_mat.shape)

    return idx1, idx2

def _merge_two_sibling_contours(
    cnt1: ContourType,
    cnt2: ContourType,
) -> ContourType:
    if not _is_clockwise(cnt1):
        cnt1 = cnt1[::-1]
    if _is_clockwise(cnt2):
        cnt2 = cnt2[::-1]

    idx1, idx2 = _get_closest_point_idx(cnt1, cnt2)

    cnt1 = np.squeeze(cnt1, axis = 1)
    cnt2 = np.squeeze(cnt2, axis = 1)

    cnt_merged = []
    cnt_merged.append(cnt1[0:idx1+1])
    cnt_merged.append(cnt2[idx2:len(cnt2)])
    cnt_merged.append(cnt2[0:idx2+1])
    cnt_merged.append(cnt1[idx1:len(cnt1)])

    cnt_merged = np.concatenate(cnt_merged, axis = 0)
    cnt_merged = np.expand_dims(cnt_merged, 1)
    
    return cnt_merged

def _merge_contours_sibling(
    cnts: List[ContourType]
) -> ContourType:
    cnt_merge = cnts[0]

    for cnt in cnts[1:]:
        cnt_merge = _merge_two_sibling_contours(cnt_merge, cnt)
    
    return cnt_merge

def _merge_contours_hierarchy(
    cnts: List[ContourType],
    hierarchies: HierarchiesType
) -> ContourType:
    cnt_group_temp = {"parent": None, "children": []}
    cnt_groups = [copy.deepcopy(cnt_group_temp) for _ in range(len(cnts))]
    for i, (cnt, hierarchy) in enumerate(zip(cnts, hierarchies[0])):
        if hierarchy[3] == -1:
            cnt_groups[i]["parent"] = cnt
        else:
            parent_cnt_idx = hierarchy[2]
            cnt_groups[parent_cnt_idx]["children"].append(cnt)
    
    cnt_groups_: List[ContourGroupType] = []
    for cnt_group in cnt_groups:
        if cnt_group["parent"] is not None:
            cnt_groups_.append(cnt_group)
    
    cnt_groups = cnt_groups_
    del cnt_groups_

    cnt_merge = cnt_group["parent"]

    for child_cnt in cnt_group["children"]:
        # cnt_merge = merge_two_contours(cnt_merge, child_cnt)

        try:
            cnt_merge = _merge_two_sibling_contours(cnt_merge, child_cnt)
        except Exception as e:
            raise(e)
    
    return cnt_merge

def merge_contours(
    cnts: List[ContourType],
    sibling_flag: bool,
    hierarchies: Union[None, HierarchiesType]
) -> ContourType:
    if sibling_flag:
        cnt_merge = _merge_contours_sibling(cnts)
        return cnt_merge
    
    assert hierarchies is not None
    
    cnt_merge = _merge_contours_hierarchy(cnts, hierarchies)
    
    return cnt_merge

def convert_contour2mask(
    cnts: Union[ContourType, List[ContourType]],
    img_hw: Tuple[int, int],
    merge_flag: bool
) -> MasksArrType:
    if not isinstance(cnt, list):
        cnts = [cnts]

    masks = []
    for cnt in cnts:
        cnt = np.squeeze(cnt, axis = 1)
        mask = np.zeros(img_hw, dtype = np.uint8)
        mask = cv2.fillPoly(mask, [cnt], color = 1)
        masks.append(mask)
    
    masks = np.stack(masks).astype(np.bool_)

    if merge_flag:
        masks = np.any(masks, axis = 0, keepdims = True)
    
    return masks

def convert_contour2poly(
    cnts: Union[ContourType, List[ContourType]]
) -> List[PolyArrType]:
    if not isinstance(cnts, list):
        cnts = [cnts]
    
    polys = []

    for cnt in cnts:
        poly = np.squeeze(cnt, axis = 1)
        polys.append(poly)
    
    return polys

def convert_contour2rle(
    cnts: Union[ContourType, List[ContourType]],
    img_hw: Tuple[int, int],
    merge_flag: bool
) -> List[RleType]:
    if not isinstance(cnts, list):
        cnts = [cnts]

    rles = []
    for cnt in cnts:
        poly_coco = cnt.flatten().tolist()
        if len(poly_coco) == 4:
            poly_coco += poly_coco[-2:]

        rle = pycocomask.frPyObjects([poly_coco], img_hw[0], img_hw[1])[0]
        rles.append(rle)
    
    if merge_flag:
        rles = [pycocomask.merge(rles)]
        
    return rles