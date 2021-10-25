# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import torch
from torch import Tensor


def atleast_2d(*tensors: Tensor):
    res = []
    for t in tensors:
        if t.ndim == 0:
            res.append(t[None, None])
        elif t.ndim == 1:
            res.append(t[:, None])
        else:
            res.append(t)
    return res if len(res) > 1 else res[0]
