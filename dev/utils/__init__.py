# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import torch
from torch import Tensor
from typing import Tuple

__all__ = ["_data_center", "atleast_2d", "one_hot"]


def _data_center(X: Tensor, y: Tensor = None) -> Tuple:
    """数据居中

    :param X: shape[N, F]
    :param y: shape[N, Out]
    :return: Tuple[X, y, X_mean, y_mean]
        X: shape[N, F]
        y: shape[N, Out]
        X_mean: shape[F]
        y_mean: shape[Out]
    """
    X_mean = torch.mean(X, dim=0)
    X = X - X_mean
    if y is not None:
        y_mean = torch.mean(y, dim=0)
        y = y - y_mean
        return X, y, X_mean, y_mean
    return X, X_mean


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


def one_hot(tensor: Tensor, n_classes: int = -1, dtype=None):
    dtype = dtype or torch.long
    device = tensor.device
    #
    if n_classes == -1:
        n_classes = torch.max(tensor) + 1
    return torch.eye(n_classes, dtype=dtype, device=device)[tensor]
