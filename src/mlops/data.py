import os
import json
import random
import shutil
from pathlib import Path
from typing import Callable, Union, Tuple


def _default_name_mapf_img2ply(
    img_name: str,
) -> str:
    img_stem = Path(img_name)
    ply_name = f"{img_stem}.ply"
    return ply_name

def raw2data(
    src_raw_root: str,
    dst_data_root: str,
    split_flag: bool,
    split_ratios: Tuple[float, float],
    split_shuffle_flag: bool,
) -> None:
    if split_flag:
        assert sum(split_ratios) == 1
        assert len(split_ratios) == 2

    batchnames = os.listdir(src_raw_root)
    batchnames.sort()

    random.seed(13)

    for bn in batchnames:
        src_batch_root = os.path.join(src_raw_root, bn)

        if not os.path.isdir(src_batch_root):
            continue
        
        if split_flag:
            dst_data_train_dir = os.path.join(dst_data_root, f"{bn}_train")
            dst_data_test_dir = os.path.join(dst_data_root, f"{bn}_test")
            os.makedirs(dst_data_train_dir, exist_ok=True)
            os.makedirs(dst_data_test_dir, exist_ok=True)
        else:
            dst_data_dir = os.path.join(dst_data_root, f"{bn}_train")
            os.makedirs(dst_data_dir, exist_ok=True)
        
        src_img_ps = []

        for root, subdirs, files in os.walk(src_batch_root):
            for file in files:
                if not file.endswith((".png", ".jpeg", ".jpg")):
                    continue
            
                src_img_name = file
                src_img_p = os.path.join(root, src_img_name)
                src_img_ps.append(src_img_p)
        
        if split_flag:
            if split_shuffle_flag:
                random.shuffle(src_img_ps)

            train_end_idx = int(len(src_img_ps) * split_ratios[0])
            src_train_img_ps = src_img_ps[:train_end_idx]
            src_test_img_ps = src_img_ps[train_end_idx:]

            data_idx = 0

            for src_img_p in src_train_img_ps:
                img_suffix = Path(src_img_p).suffix
                dst_img_name = f"{data_idx}{img_suffix}"
                dst_img_p = os.path.join(dst_data_train_dir, dst_img_name)

                shutil.copy(src_img_p, dst_img_p)

                data_idx += 1
            
            for src_img_p in src_test_img_ps:
                img_suffix = Path(src_img_p).suffix
                dst_img_name = f"{data_idx}{img_suffix}"
                dst_img_p = os.path.join(dst_data_test_dir, dst_img_name)

                shutil.copy(src_img_p, dst_img_p)

                data_idx += 1
        else:
            data_idx = 0

            for src_img_p in src_img_ps:
                img_suffix = Path(src_img_p).suffix
                dst_img_name = f"{data_idx}{img_suffix}"
                dst_img_p = os.path.join(dst_data_dir, dst_img_name)

                shutil.copy(src_img_p, dst_img_p)

                data_idx += 1

def split_data(
    src_root: str,
    dst_train_root: str,
    dst_test_root: str,
    train_test_ratios: Tuple[float, float],
    shuffle_flag: bool,
    labelme_flag: bool,
) -> None:
    assert sum(train_test_ratios) == 1

    random.seed(13)

    os.makedirs(dst_train_root, exist_ok = True)
    os.makedirs(dst_test_root, exist_ok = True)

    fns = os.listdir(src_root)
    img_names = [f for f in fns if f.endswith((".png", ".jpg", ".jpeg"))]
    img_names.sort()

    if shuffle_flag:
        random.shuffle(img_names)

    split_idx = int(len(img_names) * train_test_ratios[0])

    for i in range(0, split_idx):
        img_name = img_names[i]
        img_p = os.path.join(src_root, img_name)
        dst_img_p = os.path.join(dst_train_root, img_name)
        shutil.copy(img_p, dst_img_p)

        if not labelme_flag:
            continue

        img_stem = Path(img_name).stem
        labelme_name = f"{img_stem}.json"
        src_labelme_p = os.path.join(src_root, labelme_name)
        dst_labelme_p = os.path.join(dst_train_root, labelme_name)

        if os.path.exists(src_labelme_p):
            shutil.copy(src_labelme_p, dst_labelme_p)
    
    for i in range(split_idx, len(img_names)):
        img_name = img_names[i]
        img_p = os.path.join(src_root, img_name)
        dst_img_p = os.path.join(dst_test_root, img_name)
        shutil.copy(img_p, dst_img_p)

        if not labelme_flag:
            continue

        img_stem = Path(img_name).stem
        labelme_name = f"{img_stem}.json"
        src_labelme_p = os.path.join(src_root, labelme_name)
        dst_labelme_p = os.path.join(dst_test_root, labelme_name)

        if os.path.exists(src_labelme_p):
            shutil.copy(src_labelme_p, dst_labelme_p)