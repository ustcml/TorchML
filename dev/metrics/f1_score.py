# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 
import torch
from . import precision_recall_fscore_support
from torch import Tensor


def f1_score(y_true, y_pred, *,
             pos_label: int = 1, average: str = 'binary') -> Tensor:
    """zero_division=0

    :param y_true: shape[N]
    :param y_pred: shape[N]
    :param pos_label: 当average='binary'时生效
    :param average: {'binary', 'micro', 'macro', 'weighted', None}
    :return:
    """
    dtype = torch.float32
    y_true = torch.as_tensor(y_true, dtype=dtype)
    y_pred = torch.as_tensor(y_pred, dtype=dtype)
    #
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=pos_label, average=average)
    return f1
