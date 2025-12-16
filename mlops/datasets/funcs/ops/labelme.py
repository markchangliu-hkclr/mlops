import copy
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np

from mlops.labels.typedef.labelme import LabelmeDictType, LabelmeShapeDictType


def crop_labelme(
    labelme_dict: LabelmeDictType,
    x1y1x2y2: Tuple[int, int, int, int],
    export_img_name: str,
) -> LabelmeDictType:
    org_h = labelme_dict["imageHeight"]
    org_w = labelme_dict["imageWidth"]

    y1 = x1y1x2y2[1]
    y2 = x1y1x2y2[3]
    x1 = x1y1x2y2[0]
    x2 = x1y1x2y2[2]

    crop_y1 = y1
    crop_y2 = min(y2, org_h)
    crop_x1 = x1
    crop_x2 = min(x2, org_w)

    crop_h = crop_y2 - crop_y1
    crop_w = crop_x2 - crop_x1

    patch_shapes: List[LabelmeShapeDictType] = []

    for shape in labelme_dict["shapes"]:
        points = np.asarray(shape["points"])

        shape_min_x1 = np.min(points[:, 0]).item()
        shape_max_x2 = np.max(points[:, 0]).item()
        shape_min_y1 = np.min(points[:, 1]).item()
        shape_max_y2 = np.max(points[:, 1]).item()

        if shape_max_x2 < crop_x1 or shape_max_y2 < crop_y1:
            continue
        if shape_min_x1 > crop_x2 or shape_min_y1 > crop_y2:
            continue

        shape_patch: LabelmeShapeDictType = copy.deepcopy(shape)
        
        points_patch = np.copy(points)
        points_patch[:, 0] = points_patch[:, 0] - crop_x1
        points_patch[:, 0] = np.clip(points_patch[:, 0], 0, crop_w)
        points_patch[:, 1] = points_patch[:, 1] - crop_y1
        points_patch[:, 1] = np.clip(points_patch[:, 1], 0, crop_h)
        shape_patch["points"] = points_patch.tolist()

        patch_shapes.append(shape_patch)
    
    patch_labelme = copy.deepcopy(labelme_dict)
    patch_labelme["shapes"] = patch_shapes
    patch_labelme["imageHeight"] = crop_h
    patch_labelme["imageWidth"] = crop_w
    patch_labelme["imagePath"] = export_img_name

    return patch_labelme

def crop_labelme2patches(
    labelme_dict: LabelmeDictType, 
    patch_hw: Tuple[int, int],
) -> List[LabelmeDictType]:
    patch_labelmes: List[LabelmeDictType] = []

    org_h = labelme_dict["imageHeight"]
    org_w = labelme_dict["imageWidth"]
    org_img_name = labelme_dict["imagePath"]
    org_img_stem = Path(org_img_name).stem
    org_img_suffix = Path(org_img_name).suffix

    patch_h, patch_w = patch_hw

    num_step_h = math.ceil(org_h / patch_h)
    num_step_w = math.ceil(org_w / patch_w)

    if num_step_h == 1 and num_step_w == 1:
        patch_labelmes.append(labelme_dict)
        return patch_labelmes

    for i in range(num_step_h):
        for j in range(num_step_w):
            y1 = i * patch_h
            y2 = (i + 1) * patch_h
            x1 = j * patch_w
            x2 = (j + 1) * patch_w

            crop_y1 = y1
            crop_y2 = min(y2, org_h)
            crop_x1 = x1
            crop_x2 = min(x2, org_w)
            crop_h = crop_y2 - crop_y1
            crop_w = crop_x2 - crop_x1

            patch_shapes: List[LabelmeShapeDictType] = []

            for shape in labelme_dict["shapes"]:
                points = np.asarray(shape["points"])

                shape_min_x1 = np.min(points[:, 0]).item()
                shape_max_x2 = np.max(points[:, 0]).item()
                shape_min_y1 = np.min(points[:, 1]).item()
                shape_max_y2 = np.max(points[:, 1]).item()

                if shape_max_x2 < crop_x1 or shape_max_y2 < crop_y1:
                    continue
                if shape_min_x1 > crop_x2 or shape_min_y1 > crop_y2:
                    continue

                shape_patch: LabelmeShapeDictType = copy.deepcopy(shape)
                
                points_patch = np.copy(points)
                points_patch[:, 0] = points_patch[:, 0] - crop_x1
                points_patch[:, 0] = np.clip(points_patch[:, 0], 0, crop_w)
                points_patch[:, 1] = points_patch[:, 1] - crop_y1
                points_patch[:, 1] = np.clip(points_patch[:, 1], 0, crop_h)
                shape_patch["points"] = points_patch.tolist()

                patch_shapes.append(shape_patch)
            
            patch_labelme = copy.deepcopy(labelme_dict)
            patch_labelme["shapes"] = patch_shapes
            patch_labelme["imageHeight"] = patch_h
            patch_labelme["imageWidth"] = patch_w
            patch_labelme["imagePath"] = f"{org_img_stem}_patch{i}{j}{org_img_suffix}"

            patch_labelmes.append(patch_labelme)

    x1_center_patch = max(org_w // 2 - patch_w // 2, 0)
    x2_center_patch = min(x1_center_patch + patch_w, org_w)
    y1_center_patch = max(org_h // 2 - patch_h // 2, 0)
    y2_center_patch = min(y1_center_patch + patch_h, org_h)
    center_patch_labelme = crop_labelme(
        labelme_dict, (x1_center_patch, y1_center_patch, x2_center_patch, y2_center_patch),
        f"{org_img_stem}_patchCenter{org_img_suffix}"
    )

    patch_labelmes.append(center_patch_labelme)

    return patch_labelmes
