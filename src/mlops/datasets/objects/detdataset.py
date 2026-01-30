from typing import List, Dict, Literal

from mlops.shapes.structs.instances import Instances


class DetDataset:
    """
    Attrs
    -----
    - `img_ps: List[str]`
    - `insts_list: List[Instances]`
    - `cat_name_id_dict: Dict[str, int]`
    - `cat_id_name_dict: Dict[int, str]`
    - `shape_format: Literal["bbox", "poly"]`
    """
    def __init__(
        self,
        img_ps: List[str],
        insts_list: List[Instances],
        cat_name_id_dict: Dict[str, int],
        shape_format: Literal["bbox", "poly"]
    ) -> None:
        assert len(img_ps) == len(insts_list)
        assert shape_format in ["bbox", "poly"]

        self.img_ps = img_ps
        self.insts_list = insts_list
        self.cat_name_id_dict = cat_name_id_dict
        self.cat_id_name_dict = {v:k for f, v in cat_name_id_dict.items()}
