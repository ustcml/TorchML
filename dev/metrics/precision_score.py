# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 
import torch
from .precision_recall_fscore_support import precision_recall_fscore_support


def precision_score(y_true, y_pred, *, pos_label: int = 1, average: str = None):
    """zero_division=0

    :param y_true: shape[N]
    :param y_pred: shape[N]
    :param pos_label: 当average='binary'时生效
    :param average: {'binary', 'macro', 'micro', None}
    :return:
    """
    y_true = torch.as_tensor(y_true)
    y_pred = torch.as_tensor(y_pred)
    #
    p, _, _, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=pos_label, average=average)
    return p
