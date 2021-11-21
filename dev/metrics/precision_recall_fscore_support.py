# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 
import torch
from . import confusion_matrix
from typing import Tuple
from torch import Tensor


def precision_recall_fscore_support(
        y_true, y_pred, *,
        beta: float = 1.,
        pos_label: int = 1, average: str = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """zero_division=0. Note that zero divides(0/0 -> 0) have no warning, which is not the sklearn default.

    :param y_true: shape[N]
    :param y_pred: shape[N]
    :param beta: [0, +inf). The greater the beta, the greater the weight of recall.
        if beta=1, precision and recall have the same weight
    :param pos_label: takes effect when `average = 'binary'`
    :param average: {'binary', 'micro', 'macro', 'weighted', None}
    :return: Tuple[precision, recall, fscore, support]. support is true_sum.
        if average = None: shape[NC] [NC] [NC] [NC]
    """

    dtype = torch.float32
    y_true = torch.as_tensor(y_true, dtype=dtype)
    y_pred = torch.as_tensor(y_pred, dtype=dtype)
    #
    cm = confusion_matrix(y_true, y_pred)
    true_sum = torch.sum(cm, dim=1)  # shape[NC]
    pred_sum = torch.sum(cm, dim=0)  # shape[NC]
    tp = torch.diag(cm)  # shape[NC]
    #
    if average == "micro":  # Barely use
        true_sum = torch.sum(true_sum)
        pred_sum = torch.sum(pred_sum)
        tp = torch.sum(tp)
    elif average == "binary":
        n_classes = tp.shape[0]
        assert n_classes == 2, "n_classes: %d" % n_classes
        true_sum = true_sum[pos_label]
        pred_sum = pred_sum[pos_label]
        tp = tp[pos_label]
    #
    precision = tp / pred_sum  # shape[NC]
    recall = tp / true_sum
    beta2 = beta ** 2
    fscore = (1 + beta2) * precision * recall / (beta2 * precision + recall)
    # handle nan. with no warning
    precision = torch.nan_to_num(precision)
    recall = torch.nan_to_num(recall)
    fscore = torch.nan_to_num(fscore)
    # handle `average`
    if average == "macro":
        precision = torch.mean(precision)
        recall = torch.mean(recall)
        fscore = torch.mean(fscore)
    elif average == "weighted":
        precision = torch.sum(precision * true_sum) / torch.sum(true_sum)
        recall = torch.sum(recall * true_sum) / torch.sum(true_sum)
        fscore = torch.sum(fscore * true_sum) / torch.sum(true_sum)
    if average is not None:
        true_sum = None
    return precision, recall, fscore, true_sum
