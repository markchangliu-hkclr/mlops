import json
import os
import shutil
from pathlib import Path
from typing import Dict, Tuple, List, Union, Literal

import numpy as np
import pycocotools.mask as pycocomask

import mlops.labels.typedef.labelme as labelme_type
import mlops.labels.typedef.coco as coco_type
import mlops.shapes.convert.poly2poly as poly2poly
import mlops.shapes.convert.poly2rle as poly2rle
import mlops.labels.utils.labelme as labelme_utils


def shape_groups_to_coco_masks(
    shape_groups: labelme_type.LabelmeShapeGroupsType,
    cat_name_id_dict: Dict[str, int],
    img_hw: Tuple[int, int],
    img_id: int,
    start_ann_id: int,
    shape_type: Literal["poly", "rle"],
) -> Tuple[coco_type.CocoAnnDictType, int]:

    ### TODO: add poly merge flag

    assert shape_type in ["poly", "rle"]

    coco_anns = []
    curr_ann_id = start_ann_id

    for i, (group_id, shape_group) in enumerate(shape_groups.items()):
        if len(shape_group) == 0:
            print("Empty shape group")
            continue
        
        cat_name = shape_group[0]["label"]
        cat_id = cat_name_id_dict[cat_name]

        if shape_type == "poly":
            polys_coco = []
        elif shape_type == "rle":
            rles_coco = []

        for shape in shape_group:
            poly_labelme = shape["points"]

            if len(poly_labelme) < 2:
                print("Abnormal poly, num_points less than 2")
                continue

            if len(poly_labelme) == 2:
                poly_labelme += poly_labelme[-1]

            if shape_type == "poly":
                poly_coco = poly2poly.poly2poly_labelme2coco(poly_labelme)
                polys_coco.append(poly_coco)
            elif shape_type == "rle":
                rle_coco = poly2rle.poly2rle_labelme(poly_labelme, img_hw)
                rles_coco.append(rle_coco)

        if shape_type == "poly":
            rle = poly2rle.polys2rle_merge_coco(polys_coco, img_hw)
        elif shape_type == "rle":
            rle = pycocomask.merge(rles_coco, intersect=0)

        area = pycocomask.area(rle).item()
        bbox = pycocomask.toBbox(rle).astype(np.int32).tolist()

        coco_ann: coco_type.CocoAnnDictType = {}
        coco_ann["area"] = area
        coco_ann["bbox"] = bbox
        coco_ann["id"] = curr_ann_id
        coco_ann["iscrowd"] = 0
        coco_ann["image_id"] = img_id
        coco_ann["category_id"] = cat_id

        if shape_type == "poly":
            coco_ann["segmentation"] = polys_coco
        elif shape_type == "rle":
            for rle in rles_coco:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_ann["segmentation"] = rles_coco

        coco_anns.append(coco_ann)

        curr_ann_id += 1
    
    return coco_anns, curr_ann_id

def labelme2coco_batch(
    img_dirs: List[Union[str, os.PathLike]],
    labelme_dirs: List[Union[str, os.PathLike]],
    export_root: Union[str, os.PathLike],
    export_img_dirname: str,
    export_coco_name: str,
    cat_name_id_dict: Dict[str, int],
    shape_type: Literal["bbox", "poly", "rle"],
) -> None:
    ### TODO: bbox shape type

    assert isinstance(img_dirs, list)
    assert isinstance(labelme_dirs, list)
    assert len(img_dirs) == len(labelme_dirs)

    assert shape_type in ["bbox", "poly", "rle"]

    export_img_dir = os.path.join(export_root, "images", export_img_dirname)
    export_coco_p = os.path.join(export_root, export_coco_name)

    os.makedirs(export_img_dir, exist_ok = True)

    coco_imgs: List[coco_type.CocoImgDictType] = []
    coco_anns: List[coco_type.CocoAnnDictType] = []
    coco_cats: List[coco_type.CocoCatDictType] = []

    for cat_name, cat_id in cat_name_id_dict.items():
        coco_cat: coco_type.CocoCatDictType = {}
        coco_cat["id"] = cat_id
        coco_cat["name"] = cat_name
        coco_cats.append(coco_cat)

    curr_img_id = 0
    curr_ann_id = 0

    for img_dir, labelme_dir in zip(img_dirs, labelme_dirs):
        p_generator = labelme_utils.img_labelme_p_generator(
            img_dir, labelme_dir
        )

        for img_p, labelme_p in p_generator:
            coco_img: coco_type.CocoImgDictType = {}

            with open(labelme_p, "r") as f:
                labelme_dict: labelme_type.LabelmeDictType = json.load(f)

            img_h = labelme_dict["imageHeight"]
            img_w = labelme_dict["imageWidth"]

            img_suffix = Path(img_p).suffix
            export_img_name = f"{curr_img_id}{img_suffix}"
            export_img_p = os.path.join(export_img_dir, export_img_name)
            export_img_rel_p = str(Path(export_img_p).relative_to(export_root))

            shutil.copy(img_p, export_img_p)

            coco_img["file_name"] = export_img_rel_p
            coco_img["height"] = img_h
            coco_img["width"] = img_w
            coco_img["id"] = curr_img_id
            coco_imgs.append(coco_img)

            shape_groups = labelme_utils.get_shape_groups(labelme_dict)
            
            if shape_type in ["poly", "rle"]:
                img_coco_anns, next_ann_id = shape_groups_to_coco_masks(
                    shape_groups, cat_name_id_dict, (img_h, img_w),
                    curr_img_id, curr_ann_id, shape_type
                )
            elif shape_type in ["bbox"]:
                raise NotImplementedError

            coco_anns += img_coco_anns
        
            curr_ann_id = next_ann_id
            curr_img_id += 1
    
    coco_dict: coco_type.CocoDictType = {
        "annotations": coco_anns,
        "categories": coco_cats,
        "images": coco_imgs
    }

    with open(export_coco_p, "w") as f:
        json.dump(coco_dict, f)