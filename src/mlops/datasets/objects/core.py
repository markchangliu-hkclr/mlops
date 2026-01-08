from typing import List

from mlops.shapes.objects.instances import Instances


class DetDataset:
    def __init__(
        self,
        img_ps: List[str],
        insts_list: List[Instances],
        check_flag: bool
    ) -> None:
        self.img_ps = img_ps
        self.insts_list = insts_list
    
    
