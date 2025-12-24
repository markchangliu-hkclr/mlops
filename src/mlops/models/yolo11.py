from typing import Literal

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from mlops.shapes.insts import Insts


def infer_pth(
    img: np.ndarray,
    model: YOLO,
    score_thres: float,
    device: Literal["cpu", "cuda:0"],
    shape_type: Literal["bbox", "mask"]
) -> Insts:
    img_h, img_w = img.shape[:2]
    res = model.predict(img, verbose = False, device = device)

    res: Results = res[0].cpu().numpy()
    bboxes = res.boxes.xyxy
    scores = res.boxes.conf
    cat_ids = res.boxes.cls

    if shape_type == "mask" and res.masks is not None:
        masks = res.masks.data
    else:
        masks = None

    selected_ids = []

    for i, (cat_id, score) in enumerate(zip(cat_ids, scores)):
        # Filter res with cats
        if score < score_thres:
            continue

        selected_ids.append(i)

    scores = scores[selected_ids, ...]
    cat_ids = cat_ids[selected_ids, ...]
    bboxes = bboxes[selected_ids, ...]

    if masks is not None:
        masks = masks[selected_ids, ...]

    sorted_idx = np.argsort(scores)[::-1]
    scores = scores[sorted_idx, ...]
    cat_ids = cat_ids[sorted_idx, ...]
    bboxes = bboxes[sorted_idx, ...]

    if masks is not None: 
        masks = masks[sorted_idx, ...]

        # resize masks to img_shape
        if masks.size == 0:
            masks = np.zeros((0, img_h, img_w), dtype=np.bool_)
        elif len(masks) == 1:
            masks = np.transpose(masks, axes=[1, 2, 0])
            masks = cv2.resize(masks, (img_w, img_h))
            masks = masks[np.newaxis, ...].astype(np.bool_)
        elif len(masks) > 1:
            masks = np.transpose(masks, axes=[1, 2, 0])
            masks = cv2.resize(masks, (img_w, img_h))
            masks = np.transpose(masks, axes=[2, 0, 1]).astype(np.bool_)
    
    insts = Insts(scores, cat_ids, bboxes, masks)

    return insts