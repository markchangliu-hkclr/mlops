import os
import json
import shutil
from pathlib import Path
from typing import Dict, Literal, Union

import cv2

from mlops.labels.typedef.labelme import LabelmeDictType
from mlops.labels.convert.labelme2coco import labelme2coco_batch
from mlops.labels.convert.labelme2yolo import labelme2yolo_batch
from mlops.datasets.core.abcs import DataPreprocessorABC


def make_ds_labelme_simple(
    data_root: str,
    dataset_root: str,
    labelme_dirname: str,
    data_preprocessor: Union[None, DataPreprocessorABC]
) -> None:
    dst_labelme_dataset_root = os.path.join(dataset_root, "dataset_labelme")
    dst_train_root = os.path.join(dst_labelme_dataset_root, "train_all")
    dst_test_root = os.path.join(dst_labelme_dataset_root, "test_all")

    os.makedirs(dst_train_root, exist_ok=True)
    os.makedirs(dst_test_root, exist_ok=True)

    ds_raw_label_root = os.path.join(dataset_root, "raw_labels")
    batchnames = os.listdir(ds_raw_label_root)
    batchnames.sort()

    data_id = 0

    for bn in batchnames:
        raw_label_dir = os.path.join(ds_raw_label_root, bn, labelme_dirname)
        raw_img_dir = os.path.join(data_root, bn)

        filenames = os.listdir(raw_img_dir)
        filenames.sort()

        if bn.endswith("_test"):
            dst_root = dst_test_root
        elif bn.endswith("_train"):
            dst_root = dst_train_root
        else:
            raise NotImplementedError(f"unrecognized split {bn}")

        for filename in filenames:
            if not filename.endswith((".png", ".jpg", ".jpeg")):
                continue

            img_name = filename
            img_stem = Path(img_name).stem
            img_suffix = Path(img_name).suffix
            img_p = os.path.join(raw_img_dir, img_name)

            labelme_name = f"{img_stem}.json"
            labelme_p = os.path.join(raw_label_dir, labelme_name)

            if not os.path.exists(img_p):
                continue
            if not os.path.exists(labelme_p):
                continue
            
            if data_preprocessor is None:
                dst_img_name = f"{data_id}{img_suffix}"
                dst_labelme_name = f"{data_id}.json"
                dst_img_p = os.path.join(dst_root, dst_img_name)
                dst_labelme_p = os.path.join(dst_root, dst_labelme_name)

                shutil.copy(img_p, dst_img_p)
                shutil.copy(labelme_p, dst_labelme_p)

                with open(dst_labelme_p, "r") as f:
                    labelme_dict: LabelmeDictType = json.load(f)
                    labelme_dict["imageData"] = None
                    labelme_dict["imagePath"] = dst_img_name
                
                with open(dst_labelme_p, "w") as f:
                    json.dump(labelme_dict, f)

                data_id += 1
            
            else:
                img = cv2.imread(img_p)
                with open(labelme_p, "r") as f:
                    labelme_dict = json.load(f)

                if data_preprocessor.output_type == "single":
                    img_out, labelme_out = data_preprocessor.process_single_output(
                        img, labelme_dict
                    )
                    imgs_out = [img_out]
                    labelmes_out = [labelme_out]
                else:
                    imgs_out, labelmes_out = data_preprocessor.process_multi_outputs(
                        img, labelme_dict
                    )
                
                for img_out, labelme_out in zip(imgs_out, labelmes_out):
                    dst_img_name = f"{data_id}{img_suffix}"
                    dst_labelme_name = f"{data_id}.json"
                    dst_img_p = os.path.join(dst_root, dst_img_name)
                    dst_labelme_p = os.path.join(dst_root, dst_labelme_name)

                    cv2.imwrite(dst_img_p, img_out)
                    
                    labelme_out["imageData"] = None
                    labelme_out["imagePath"] = dst_img_name
                    
                    with open(dst_labelme_p, "w") as f:
                        json.dump(labelme_out, f)

                    data_id += 1

def convert_ds_labelme2yolo(
    dataset_root: str,
    cat_name_id_dict: Dict[str, int],
    labelme_dirname: str,
    shape_type: Literal["bbox", "poly"]
) -> None:
    labelme_root = os.path.join(dataset_root, "dataset_labelme")
    yolo_root = os.path.join(dataset_root, "dataset_yolo")
    splits = os.listdir(labelme_root)
    splits = [s for s in splits if s.startswith(("train_", "test_"))]
    splits = [s for s in splits if os.path.isdir(os.path.join(labelme_root, s))]

    for split in splits:
        img_dir = os.path.join(labelme_root, split)
        labelme_dir = os.path.join(labelme_root, split)

        labelme2yolo_batch(
            [img_dir], [labelme_dir], yolo_root,
            split, cat_name_id_dict, shape_type
        )

def convert_ds_labelme2coco(
    dataset_root: str,
    cat_name_id_dict: Dict[str, int],
    labelme_dirname: str,
    shape_type: Literal["bbox", "poly", "rle"]
) -> None:
    labelme_root = os.path.join(dataset_root, "dataset_labelme")
    coco_root = os.path.join(dataset_root, "dataset_coco")
    splits = os.listdir(labelme_root)
    splits = [s for s in splits if s.startswith(("train_", "test_"))]
    splits = [s for s in splits if os.path.isdir(os.path.join(labelme_root, s))]

    for split in splits:
        img_dir = os.path.join(labelme_root, split)
        labelme_dir = os.path.join(labelme_root, split)

        export_coco_name = f"{split}.json"

        labelme2coco_batch(
            [img_dir], [labelme_dir], coco_root, split, 
            export_coco_name, cat_name_id_dict, shape_type
        )