import cv2
import numpy as np
import numpy.typing as npt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from mlops.models.base import BaseModel
from mlops.shapes.typedef.bboxes import BBoxesXYXYArrType
from mlops.shapes.typedef.masks import MasksType


class Sam2Model(BaseModel):
    def __init__(
        self,
        ckpt_p: str,
        cfg_p: str,
        device: str
    ) -> None:
        assert device.startswith("cuda:")

        ckpt_p = "/" + ckpt_p
        cfg_p = "/" + cfg_p

        sam2_model = build_sam2(cfg_p, ckpt_p, device = device)
        predictor = SAM2ImagePredictor(sam2_model)

        self.predictor = predictor
    
    def infer_masks_given_bboxes(
        self, 
        bgr: npt.NDArray[np.uint8], 
        bboxes: BBoxesXYXYArrType
    ) -> MasksType:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)

        labels = np.ones(len(bboxes))

        masks, scores, logits = self.predictor.predict(
            point_coords = None,
            point_labels = None,
            box = bboxes,
            multimask_output = False
        )

        if len(masks.shape) == 4:
            masks = np.squeeze(masks, 1).astype(np.bool_)
        elif len(masks.shape) == 3:
            masks = masks.astype(np.bool_)
        else:
            raise RuntimeError

        return masks