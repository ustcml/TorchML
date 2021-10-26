# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import torch
from torch import Tensor


def r2_score(
        y_true, y_pred) -> Tensor:
    """

    :param y_true: shape[N]
    :param y_pred: shape[N]
    :return:
    """
    dtype = torch.float32
    y_true = torch.as_tensor(y_true, dtype=dtype)
    y_pred = torch.as_tensor(y_pred, dtype=dtype)
    #
    u = torch.sum((y_true - y_pred) ** 2, dim=0)
    y_true_mean = torch.mean(y_true, dim=0)
    v = torch.sum((y_true - y_true_mean) ** 2, dim=0)
    return torch.mean(1 - u / v)
