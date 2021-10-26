# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import torch
from torch import Tensor


def accuracy_score(
        y_true, y_pred, *,
        normalize: bool = True) -> Tensor:
    """

    :param y_true: shape[N]
    :param y_pred: shape[N]
    :param normalize: True: 返回float; False: 返回int
    :return:
    """
    dtype = torch.float32
    y_true = torch.as_tensor(y_true, dtype=dtype)
    y_pred = torch.as_tensor(y_pred, dtype=dtype)
    #
    N = y_true.shape[0]
    res = torch.count_nonzero(y_true == y_pred)
    return res / N if normalize else res
