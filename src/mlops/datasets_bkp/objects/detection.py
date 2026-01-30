from typing import List, Dict, Literal

from mlops.shapes.objects.instances import Instances


class DetDataset:
    def __init__(
        self,
        img_ps: List[str],
        insts_list: List[Instances],
        cat_id_name_dict: Dict[int, str],
    ) -> None:
        self.img_ps = img_ps
        self.insts_list = insts_list
        self.cat_id_name_dict = cat_id_name_dict
        self.cat_name_id_dict = {v:k for k, v in cat_id_name_dict.items()}
    
    @classmethod
    def init_from_labelme(
        cls,
        labelme_dirs: List[str],
        img_dirs: List[str],
        cat_name_id_dict: Dict[str, int],
        label_format: Literal["bbox", "mask"]
    ) -> "DetDataset":
        pass
    
    @classmethod
    def init_from_coco(
        cls,
        coco_p: str,
        img_prefix: str,
    ) -> "DetDataset":
        pass
    
    @classmethod
    def init_from_yolo(
        cls,
        img_root: str,
        label_root: str
    ) -> "DetDataset":
        pass

    def dump_labelme(
        self,
        img_dir: str,
        labelme_dir: str
    ) -> None:
        pass

    def dump_coco(
        self,
        img_dir: str,
        json_p: str
    ) -> None:
        pass

    def dump_yolo(
        self,
        img_root: str,
        label_root: str,
    ) -> None:
        pass

    def concat(
        self,
        dataset_list: List["DetDataset"],
        update_flag: bool
    ) -> "DetDataset":
        pass