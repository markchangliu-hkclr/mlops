import json
import shutil
import os
from pathlib import Path
from typing import List, Dict, Tuple, Union, Literal

import numpy as np

import mlops.labels.typedef.labelme as labelme_type
import mlops.labels.typedef.yolo as yolo_type
import mlops.shapes.merge.cnts as cnt_merge
import mlops.shapes.convert.poly2cnt as poly_cvt
import mlops.shapes.convert.cnt2poly as cnt_cvt
import mlops.labels.utils.labelme as labelme_utils


def shape_groups_to_yolo_poly(
    shape_groups: labelme_type.LabelmeShapeGroupsType,
    cat_name_id_dict: Dict[str, int],
    img_hw: Tuple[int, int]
) -> yolo_type.YoloPolyLabelsType:
    yolo_labels = []

    # Merge shapes with the same group_id
    for group_id, shape_group in shape_groups.items():
        if len(shape_group) == 0:
            print("Empty shape group")
            continue
        if len(shape_group) == 1:
            # Non-occluded instance, 1 poly
            poly_labelme = shape_group[0]["points"]
            cnt = poly_cvt.poly2cnt_labelme(poly_labelme)
        else:
            # Occluded instance, 
            # 1 bbox + multiple polys, or multiple polys
            cnts = []

            for shape in shape_group:
                if shape["shape_type"] == "rectangle":
                    continue

                poly_labelme = shape["points"]
                cnt = poly_cvt.poly2cnt_labelme(poly_labelme)
            
            if len(cnts) == 0:
                continue
            
            cnt = cnt_merge.merge_contours_sibling(cnts)

        if len(cnt) == 0:
            print("Empty contour")
            continue

        poly_yolo = cnt_cvt.cnt2poly_yolo(cnt, img_hw)

        cat_name = shape_group[0]["label"]
        cat_id = cat_name_id_dict[cat_name]

        yolo_label = [cat_id] + poly_yolo
        yolo_labels.append(yolo_label)

    return yolo_labels

def shapes_to_yolo_bbox(
    shapes: List[labelme_type.LabelmeShapeDictType],
    cat_name_id_dict: Dict[str, int],
    img_hw: Tuple[int, int]
) -> yolo_type.YoloBBoxLabelsType:
    yolo_labels = []

    for shape in shapes:
        if shape["shape_type"] != "rectangle":
            continue

        points = np.asarray(shape["points"]).flatten().tolist()

        if len(points) != 4:
            print("Abnormal bbox, num_points != 4")
        
        x1 = min(points[0], points[2])
        y1 = min(points[1], points[3])
        x2 = max(points[0], points[2])
        y2 = max(points[1], points[3])
        x_ctr_norm = (x1 + x2) / 2 / img_hw[1]
        y_ctr_norm = (y1 + y2) / 2 / img_hw[0]
        w_norm = (x2 - x1) / img_hw[1]
        h_norm = (y2 - y1) / img_hw[0]

        cat_id = cat_name_id_dict[shape["label"]]

        yolo_bbox = [cat_id, x_ctr_norm, y_ctr_norm, w_norm, h_norm]
        yolo_labels.append(yolo_bbox)

    return yolo_labels

def labelme2yolo_file(
    img_p: Union[str, os.PathLike],
    labelme_p: Union[str, os.PathLike],
    export_img_p: Union[str, os.PathLike],
    export_label_p: Union[str, os.PathLike],
    cat_name_id_dict: Dict[str, int],
    shape_type: Literal["bbox", "poly"]
) -> None:
    assert shape_type in ["bbox", "poly"]

    shutil.copy(img_p, export_img_p)

    with open(labelme_p, "r") as f:
        labelme_dict: labelme_type.LabelmeDictType = json.load(f)

    img_h = labelme_dict["imageHeight"]
    img_w = labelme_dict["imageWidth"]

    if shape_type == "poly":
        shape_groups = labelme_utils.get_shape_groups(labelme_dict)
        yolo_labels = shape_groups_to_yolo_poly(
            shape_groups, cat_name_id_dict, (img_h, img_w)
        )
    elif shape_type == "bbox":
        shapes = labelme_dict["shapes"]
        yolo_labels = shapes_to_yolo_bbox(
            shapes, cat_name_id_dict, (img_h, img_w)
        )

    with open(export_label_p, "w") as f:
        for l in yolo_labels:
            l = [str(e) for e in l]
            l_txt = " ".join(l)
            l_txt = f"{l_txt}\n"
            f.write(l_txt)

def labelme2yolo_batch(
    img_dirs: List[Union[str, os.PathLike]],
    labelme_dirs: List[Union[str, os.PathLike]],
    export_root: Union[str, os.PathLike],
    export_foldername: str,
    cat_name_id_dict: Dict[str, int],
    shape_type: Literal["bbox", "poly"]
) -> None:
    assert isinstance(img_dirs, list)
    assert isinstance(labelme_dirs, list)
    assert len(img_dirs) == len(labelme_dirs)

    assert shape_type in ["bbox", "poly"]

    export_img_dir = os.path.join(export_root, "images", export_foldername)
    export_label_dir = os.path.join(export_root, "labels", export_foldername)

    os.makedirs(export_img_dir, exist_ok = True)
    os.makedirs(export_label_dir, exist_ok = True)

    curr_img_id = 0

    for img_dir, labelme_dir in zip(img_dirs, labelme_dirs):
        p_generator = labelme_utils.img_labelme_p_generator(
            img_dir, labelme_dir
        )

        for img_p, labelme_p in p_generator:
            img_suffix = Path(img_p).suffix
            export_img_name = f"{curr_img_id}{img_suffix}"
            export_img_p = os.path.join(export_img_dir, export_img_name)

            export_label_name = f"{curr_img_id}.txt"
            export_label_p = os.path.join(export_label_dir, export_label_name)

            labelme2yolo_file(
                img_p, labelme_p, export_img_p, export_label_p, 
                cat_name_id_dict, shape_type
            )

            curr_img_id += 1

def labelme2yolo_batch_split(
    img_dirs: Union[str, os.PathLike],
    labelme_dirs: Union[str, os.PathLike],
    export_root: Union[str, os.PathLike],
    export_foldernames: List[str],
    splits_kwords: List[List[str]],
    cat_name_id_dict: Dict[str, int],
    shape_type: Literal["bbox", "poly"]
) -> None:
    assert isinstance(img_dirs, list)
    assert isinstance(labelme_dirs, list)
    assert len(img_dirs) == len(labelme_dirs)

    assert shape_type in ["bbox", "poly"]

    splits_convert_kwargs = []

    for kws, foldername in zip(splits_kwords, export_foldernames):
        split_convert_kwargs = {
            "img_dirs": [],
            "labelme_dirs": [],
            "export_root": export_root,
            "export_foldername": foldername,
            "cat_name_id_dict": cat_name_id_dict,
            "shape_type": shape_type
        }

        for img_dir, labelme_dir in zip(img_dirs, labelme_dirs):
            include_flags = []
            
            for kw in kws:
                include_flag = False

                if "!" in kw:
                    tokens_exclude = kw.split("!")
                    tokens_exclude = [t.strip() for t in tokens_exclude if len(t) > 0]
                else:
                    tokens_exclude = []

                if "|" in kw:
                    tokens = kw.split("|")
                    tokens = [t.strip() for t in tokens if len(t) > 0]
                else:
                    tokens = [kw]

                for token in tokens:
                    if token in img_dir:
                        include_flag = True
                        break
                
                for token in tokens_exclude:
                    if token in img_dir:
                        include_flag = False
                        break
                
                include_flags.append(include_flag)
            
            if not all(include_flags):
                continue

            split_convert_kwargs["img_dirs"].append(img_dir)
            split_convert_kwargs["labelme_dirs"].append(labelme_dir)

            splits_convert_kwargs.append(split_convert_kwargs)
    
    for kwargs in splits_convert_kwargs:
        labelme2yolo_batch(**kwargs)