# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import torch
from torch import Tensor
from typing import Tuple


def _binary_clf_curve(
        y_true, y_score) -> Tuple[Tensor, Tensor, Tensor]:
    """计算fps, tps, thresholds

    :param y_true: shape[N]
    :param y_score: shape[N]
    :return: Tuple[fps, tps, thresholds]
        fps: shape[T]. 递增. T: n_thresholds
        tps: shape[T]. 递增
        thresholds: shape[T]. 递减
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
    """PR图中: y轴为P, x轴为R

    :param y_true: shape[N]
    :param probas_pred: shape[N]
    :return: Tuple[precision, recall, thresholds]
        precision: shape[T+1]. 趋势递增. T: n_thresholds
        recall: shape[T+1]. 递减
        thresholds: shape[T]. 递增
    """
    dtype = torch.float32
    y_true = torch.as_tensor(y_true, dtype=dtype)
    probas_pred = torch.as_tensor(probas_pred, dtype=dtype)
    device = y_true.device
    #
    fp, tp, thresholds = _binary_clf_curve(y_true, probas_pred)  # shape[T] [T] [T]
    # tp[-1] = m+, fp[-1] = m-. 因为最低阀值包含所有true_pos, true_neg
    precision = tp / (tp + fp)
    recall = tp / tp[-1]
    # recall最后不提升/变化的略去. 因为对AP不贡献
    last_ind = int(torch.searchsorted(tp, tp[-1])) + 1
    # 加入pr图(0, 1)点. 并进行reverse，使recall递减(同sklearn实现)
    one = torch.tensor([1.], device=device)
    zero = torch.tensor([0.], device=device)
    # reverse and cat
    precision = torch.cat([torch.flip(precision[:last_ind], (0,)), one])
    recall = torch.cat([torch.flip(recall[:last_ind], (0,)), zero])
    thresholds = torch.flip(thresholds[:last_ind], (0,))
    return precision, recall, thresholds
