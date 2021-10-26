# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import torch
from torch import Tensor
from typing import Union


def accuracy_score(
        y_true, y_pred, *,
        normalize: bool = True) -> Union[float, int]:
    """

    :param y_true: shape[N]
    :param y_pred: shape[N]
    :param normalize: True: 返回float; False: 返回int
    :return:
    """
    y_true = torch.as_tensor(y_true)
    y_pred = torch.as_tensor(y_pred)
    #
    N = y_true.shape[0]
    res = torch.count_nonzero(y_true == y_pred).item()
    return res / N if normalize else res
