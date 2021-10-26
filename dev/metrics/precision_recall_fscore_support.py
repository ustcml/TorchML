# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 
import torch
from . import confusion_matrix
from typing import Tuple


def precision_recall_fscore_support(
        y_true, y_pred, *,
        beta: float = 1.,
        pos_label: int = 1, average: str = None,
) -> Tuple:
    """zero_division=0. 注意零除no warning, 不是sklearn默认值.

    :param y_true: shape[N]
    :param y_pred: shape[N]
    :param beta: [0, +inf). beta越大 recall的权重越大.
        beta=1 precision和recall权重一致
    :param pos_label: 当average='binary'时生效
    :param average: {'binary', 'macro', 'micro', None}
    :return: Tuple[precision, recall, fscore, support]. support即true_sum
    """
    y_true = torch.as_tensor(y_true)
    y_pred = torch.as_tensor(y_pred)
    #
    cm = confusion_matrix(y_true, y_pred)
    true_sum = torch.sum(cm, dim=1)  # shape[NC]
    pred_sum = torch.sum(cm, dim=0)  # shape[NC]
    tp = torch.diag(cm)  # shape[NC]
    #
    if average == "micro":  # 基本不会用
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
    # 处理nan. no warning
    precision = torch.nan_to_num(precision)
    recall = torch.nan_to_num(recall)
    fscore = torch.nan_to_num(fscore)
    # 处理average
    if average == "macro":
        precision = torch.mean(precision)
        recall = torch.mean(recall)
        fscore = torch.mean(fscore)
    if average is not None:
        precision = precision.item()
        recall = recall.item()
        fscore = fscore.item()
        true_sum = None
    return precision, recall, fscore, true_sum
