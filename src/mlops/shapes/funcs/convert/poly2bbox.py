import numpy as np

import mlops.shapes.typing.bbox as bbox_type
import mlops.shapes.typing.poly as poly_type


__all__ = [
    "poly2bbox_coco",
    "polys2bbox_coco"
]


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