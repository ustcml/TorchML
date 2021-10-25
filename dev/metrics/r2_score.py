# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import torch


def r2_score(
        y_true, y_pred) -> float:
    """

    :param y_true: shape[N]
    :param y_pred: shape[N]
    :return: scalar
    """
    y_true = torch.as_tensor(y_true)
    y_pred = torch.as_tensor(y_pred)

    u = torch.sum((y_true - y_pred) ** 2, dim=0)
    y_true_mean = torch.mean(y_true, dim=0)
    v = torch.sum((y_true - y_true_mean) ** 2, dim=0)
    return torch.mean(1 - u / v).item()
