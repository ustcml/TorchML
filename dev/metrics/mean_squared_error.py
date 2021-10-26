# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 
import torch
from torch import Tensor
from typing import Union


def mean_squared_error(
        y_true, y_pred, *,
        multioutput="uniform_average",
        squared: bool = True, return_tensor: bool = False) -> Union[float, Tensor]:
    """

    :param y_true: shape[N, Out] or shape[N]
    :param y_pred: shape[N, Out] or shape[N]
    :param multioutput: {'raw_values', 'uniform_average'} or shape[Out]
    :param squared: True: MSE; False: RMSE
    :param return_tensor: 是否返回Tensor
    :return:
    """
    y_true = torch.as_tensor(y_true)
    y_pred = torch.as_tensor(y_pred)
    #
    res = torch.mean((y_true - y_pred) ** 2, dim=0)
    if squared is False:
        res = torch.sqrt(res)  # rmse
    #
    if isinstance(multioutput, str):
        # 'raw_values'不做处理
        if multioutput == "uniform_average":
            res = torch.mean(res)
    else:
        # multioutput is shape[Out]
        res = torch.sum(res * multioutput)
    return res if return_tensor else res.item()
