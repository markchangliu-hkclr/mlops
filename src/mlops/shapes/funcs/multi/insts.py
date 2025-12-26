from typing import List

from mlops.shapes.objects.insts import Insts


__all__ = [
    "concat"
]

import numpy as np

np.sum()

def concat(
    insts_list: List[Insts]
) -> Insts:
    new_insts = insts_list[0].concat(insts_list[1:])
    return new_insts