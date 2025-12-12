import os
import json
import random
import shutil
from pathlib import Path
from typing import Callable, Union, Tuple

import cv2

from mlops.labels.typedef.labelme import LabelmeDictType


def _default_name_mapf_img2ply(
    img_name: str,
) -> str:
    img_stem = Path(img_name)
    ply_name = f"{img_stem}.ply"
    return ply_name

def raw2data(
    src_raw_root: str,
    dst_data_root: str,
    labelme_flag: bool,
) -> None:
    data_id = 0

    os.makedirs(dst_data_root, exist_ok = True)

    for root, subdirs, files in os.walk(src_raw_root):
        for f in files:
            if not f.endswith((".png", ".jpeg", ".jpg")):
                continue

            src_img_name = f
            src_img_suffix = Path(src_img_name).suffix
            src_img_stem = Path(src_img_name).stem
            src_img_p = os.path.join(root, src_img_name)

            dst_img_name = f"img{data_id}{src_img_suffix}"
            dst_img_p = os.path.join(dst_data_root, dst_img_name)

            shutil.copy(src_img_p, dst_img_p)

            if not labelme_flag:
                data_id += 1
                continue

            src_labelme_name = f"{src_img_stem}.json"
            src_labelme_p = os.path.join(root, src_labelme_name)
            dst_labelme_name = f"img{data_id}.json"
            dst_labelme_p = os.path.join(dst_data_root, dst_labelme_name)

            if not os.path.exists(src_labelme_p):
                data_id += 1
                continue

            shutil.copy(src_labelme_p, dst_labelme_p)

            with open(dst_labelme_p, "r") as f:
                dst_labelme: LabelmeDictType = json.load(f)

            dst_labelme["imageData"] = None
            dst_labelme["imagePath"] = dst_img_name

            with open(dst_labelme_p, "w") as f:
                json.dump(dst_labelme, f)
            
            data_id += 1

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

def video2imgs(
    video_p: str,
    img_dir: str,
    save_frame_period: int
) -> None:
    """
    Args
    - `save_frame_period`: `int`, save 1 frame for every `save_frame_period` frames
    """
    os.makedirs(img_dir, exist_ok = True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_p)
    
    if not cap.isOpened():
        print(f"can not open video '{video_p}'")
        return
    
    # 获取视频基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
    duration = total_frames / fps  # 视频总时长（秒）
    
    print(f"video_p: {video_p}")
    print(f"video info: {fps:.2f}fps, total {total_frames} frames, duration {duration:.2f}s")
    
    frame_count = 0
    start_frame = 0
    saved_count = 0
    end_frame = total_frames
    current_frame = start_frame

    print("extracting frames...")
    
    while current_frame <= end_frame:
        ret, frame = cap.read()
        
        if not ret:
            print("no more frames, video ends")
            break
        
        # 每隔frame_interval帧保存一张图片
        if frame_count % save_frame_period == 0:
            # 生成文件名（使用帧编号，便于排序）
            frame_filename = f"frame_{current_frame:06d}.jpg"
            output_path = os.path.join(img_dir, frame_filename)
            
            # 保存帧为图片
            success = cv2.imwrite(output_path, frame)
            
            if success:
                saved_count += 1
                if saved_count % 10 == 0:  # 每保存10张图片打印一次进度
                    print(f"saved {saved_count} frames, progress: {current_frame}/{total_frames}")
            else:
                print(f"save failed: {output_path}")
        
        frame_count += 1
        current_frame += 1
        
        # 跳过中间帧，提高处理速度
        if save_frame_period > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    
    # 释放资源
    cap.release()
    print(f"complete, {saved_count} imgs saved at {img_dir}")
    print()
    