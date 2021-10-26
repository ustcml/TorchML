# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 
import torch
from .precision_recall_fscore_support import precision_recall_fscore_support
from torch import Tensor


def fbeta_score(y_true, y_pred, *,
                beta: float = 1., pos_label: int = 1, average: str = 'binary') -> Tensor:
    """zero_division=0

    :param y_true: shape[N]
    :param y_pred: shape[N]
    :param beta: [0, +inf). beta越大 recall的权重越大.
        beta=1 precision和recall权重一致
    :param pos_label: 当average='binary'时生效
    :param average: {'binary', 'micro', 'macro', 'weighted', None}
    :return:
    """
    dtype = torch.float32
    y_true = torch.as_tensor(y_true, dtype=dtype)
    y_pred = torch.as_tensor(y_pred, dtype=dtype)
    #
    _, _, fbeta, _ = precision_recall_fscore_support(
        y_true, y_pred, beta=beta, pos_label=pos_label, average=average)
    return fbeta
