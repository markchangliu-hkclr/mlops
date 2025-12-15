import json
import os
from pathlib import Path
import shutil
from typing import Union, Dict, Literal, List

import cv2
from mlops.shapes.convert.poly2poly import poly2poly_yolo2labelme
from mlops.shapes.convert.bbox2bbox import bbox2bbox_yolo2labelme
from mlops.labels.typedef.labelme import LabelmeDictType, LabelmeShapeDictType


def yolo2labelme_file(
    img_p: str,
    yolo_ann_p: str,
    export_img_p: str,
    export_labelme_p: str,
    cat_id_name_dict: Dict[int, str],
    shape_type: Literal["bbox", "poly"]
) -> None:
    assert shape_type in ["bbox", "poly"]

    # if shape_type == "bbox":
    #     raise NotImplementedError

    if shape_type == "poly":
        labelme_shape_type = "polygon"
    else:
        labelme_shape_type = "rectangle"

    cat_ids = []
    anns_yolo = []

    img = cv2.imread(img_p)
    img_h, img_w = img.shape[:2]

    with open(yolo_ann_p, "r") as f:
        for line in f:
            line = line.strip()
            line = line.split(" ")
            cat_id = int(line[0])
            ann_yolo = line[1:]
            ann_yolo = [float(p) for p in ann_yolo]

            cat_ids.append(cat_id)
            anns_yolo.append(ann_yolo)
    
    labelme_dict: LabelmeDictType = {}
    labelme_dict["version"] = "5.1.1"
    labelme_dict["flags"] = {}
    labelme_dict["imageData"] = None
    labelme_dict["imageHeight"] = img_h
    labelme_dict["imageWidth"] = img_w
    labelme_dict["imagePath"] = Path(export_img_p).name
    labelme_dict["shapes"] = []

    for cat_id, ann_yolo in zip(cat_ids, anns_yolo):
        if cat_id not in cat_id_name_dict.keys():
            continue
        
        if shape_type == "poly":
            shape_labelme = poly2poly_yolo2labelme(ann_yolo, (img_h, img_w))
        else:
            shape_labelme = bbox2bbox_yolo2labelme(ann_yolo, (img_h, img_w))

        labelme_shape_dict: LabelmeShapeDictType = {}
        labelme_shape_dict["flags"] = {}
        labelme_shape_dict["group_id"] = None
        labelme_shape_dict["label"] = cat_id_name_dict[cat_id]
        labelme_shape_dict["points"] = shape_labelme
        labelme_shape_dict["shape_type"] = labelme_shape_type

        labelme_dict["shapes"].append(labelme_shape_dict)
    
    with open(export_labelme_p, "w") as f:
        json.dump(labelme_dict, f)
    
    shutil.copy(img_p, export_img_p)

def yolo2labelme_batch(
    img_dirs: List[str],
    yolo_ann_dirs: List[str],
    export_img_dirs: List[str],
    export_labelme_dirs: List[str],
    cat_id_name_dict: Dict[int, str],
    shape_type: Literal["bbox", "poly"]
) -> None:
    assert isinstance(img_dirs, list)
    assert isinstance(yolo_ann_dirs, list)
    assert isinstance(export_img_dirs, list)
    assert isinstance(export_labelme_dirs, list)

    assert len(img_dirs) == len(yolo_ann_dirs) == len(export_img_dirs) \
        == len(export_labelme_dirs)

    for i in range(len(img_dirs)):
        img_dir = img_dirs[i]
        yolo_ann_dir = yolo_ann_dirs[i]
        export_img_dir = export_img_dirs[i]
        export_labelme_dir = export_labelme_dirs[i]

        os.makedirs(export_img_dir, exist_ok=True)
        os.makedirs(export_labelme_dir, exist_ok=True)

        fns = os.listdir(img_dir)
        fns.sort()

        for fn in fns:
            if not fn.endswith((".png", ".jpg", ".jpeg")):
                continue
            
            img_p = os.path.join(img_dir, fn)
            img_stem = Path(fn).stem
            yolo_ann_p = os.path.join(yolo_ann_dir, f"{img_stem}.txt")

            if not os.path.exists(yolo_ann_p):
                continue

            export_img_p = os.path.join(export_img_dir, fn)
            export_labelme_p = os.path.join(export_labelme_dir, f"{img_stem}.json")

            yolo2labelme_file(
                img_p, yolo_ann_p, export_img_p, export_labelme_p,
                cat_id_name_dict, shape_type
            )