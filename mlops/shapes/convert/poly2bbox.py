import numpy as np

import mlops.shapes.typedef.bboxes as bbox_type
import mlops.shapes.typedef.polys as poly_type


def poly2bbox_coco(
    poly: poly_type.PolyCocoType
) -> bbox_type.BBoxCocoType:
    poly = np.asarray(poly).reshape(-1, 2)
    x1 = int(np.min(poly[:, 0]).item())
    x2 = int(np.max(poly[:, 0]).item())
    y1 = int(np.min(poly[:, 1]).item())
    y2 = int(np.max(poly[:, 1]).item())

    w = x2 - x1
    h = y2 - y1

    bbox = [x1, y1, 2, h]

    return bbox

def polys2bbox_coco(
    polys: poly_type.PolysCocoType
) -> bbox_type.BBoxCocoType:
    polys = [np.asarray(p).reshape(-1, 2) for p in polys]
    polys = np.concatenate(polys, axis = 0)

    x1 = int(np.min(polys[:, 0]).item())
    x2 = int(np.max(polys[:, 0]).item())
    y1 = int(np.min(polys[:, 1]).item())
    y2 = int(np.max(polys[:, 1]).item())

    w = x2 - x1
    h = y2 - y1

    bbox = [x1, y1, 2, h]

    return bbox

def poly2bbox_labelme(
    poly: poly_type.PolyLabelmeType
) -> bbox_type.BBoxLabelmeType:
    poly = np.asarray(poly)
    x1 = int(np.min(poly[:, 0]).item())
    x2 = int(np.max(poly[:, 0]).item())
    y1 = int(np.min(poly[:, 1]).item())
    y2 = int(np.max(poly[:, 1]).item())

    bbox = [[x1, y1], [x2, y2]]

    return bbox

def polys2bbox_labelme(
    polys: poly_type.PolysLabelmeType
) -> bbox_type.BBoxLabelmeType:
    polys = np.asarray(polys).reshape(-1, 2)
    x1 = int(np.min(polys[:, 0]).item())
    x2 = int(np.max(polys[:, 0]).item())
    y1 = int(np.min(polys[:, 1]).item())
    y2 = int(np.max(polys[:, 1]).item())

    bbox = [[x1, y1], [x2, y2]]

    return bbox