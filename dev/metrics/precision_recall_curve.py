# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import torch
from torch import Tensor
from typing import Tuple


def _binary_clf_curve(
        y_true, y_score) -> Tuple[Tensor, Tensor, Tensor]:
    """

    :param y_true: shape[N]
    :param y_score: shape[N]
    :return: tp, fp, thresholds. shape[T], shape[T], shape[T]. T: n_thresholds
        fp: 递增
        tp: 递增
        thresholds: 递减
    """
    device = y_score.device
    # 降序score排序
    desc_score_idxs = torch.argsort(y_score, descending=True)
    y_score = y_score[desc_score_idxs]
    y_true = y_true[desc_score_idxs]
    # score跃迁的索引为threshold_idx
    ys_diff = torch.diff(y_score, append=torch.tensor([y_score[-1] - 1], device=device))  # shape[N]
    threshold_idxs = torch.where(ys_diff)[0]  # shape[T]
    #
    tps = torch.cumsum(y_true, dim=0)[threshold_idxs]
    # or:
    # fps = torch.cumsum(1 - y_true, dim=0)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # 更快

    return fps, tps, y_score[threshold_idxs]


def precision_recall_curve(
        y_true, probas_pred) -> Tuple[Tensor, Tensor, Tensor]:
    """PR图中: y轴为p, x轴为r

    :param y_true: shape[N]
    :param probas_pred: shape[N]
    :return: precision, recall, thresholds: shape[T+1], shape[T+1], shape[T]. T: n_thresholds
        precision: 趋势递增
        recall: 递减
        thresholds: 递增
    """
    dtype = torch.float32
    device = y_true.device
    y_true = torch.as_tensor(y_true, dtype=dtype)
    probas_pred = torch.as_tensor(probas_pred, dtype=dtype)
    #
    fp, tp, thresholds = _binary_clf_curve(y_true, probas_pred)  # shape[T] [T] [T]
    #
    precision = tp / (tp + fp)
    # tp[-1] = m+, fp[-1] = m-. 最低阀值包含所有true_pos, true_neg
    recall = tp / tp[-1]
    # recall最后不提升/变化的略去. 因为对AP不贡献
    last_ind = int(torch.searchsorted(tp, tp[-1])) + 1
    # 加入pr图(0, 1)点. 并进行reverse，使recall递减(同sklearn实现)
    precision = torch.cat([torch.flip(precision[:last_ind], (0,)), torch.tensor([1], device=device)])
    recall = torch.cat([torch.flip(recall[:last_ind], (0,)), torch.tensor([0], device=device)])
    thresholds = torch.flip(thresholds[:last_ind], (0,))
    return precision, recall, thresholds
