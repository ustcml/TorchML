# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

from torch import Tensor
import torch
from . import precision_recall_curve


def average_precision_score(y_true, y_score) -> Tensor:
    """

    :param y_true: shape[N]
    :param y_score: shape[N]
    :return:
    """
    dtype = torch.float32
    y_true = torch.as_tensor(y_true, dtype=dtype)
    y_score = torch.as_tensor(y_score, dtype=dtype)
    # precision趋势递增, recall递减, thresholds递增
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    # 求面积. 不含precision[-1] = 1.
    return -torch.sum(torch.diff(recall) * precision[:-1])
