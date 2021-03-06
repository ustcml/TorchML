# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import torch
from torch import Tensor
from typing import Tuple


def _binary_clf_curve(
        y_true, y_score) -> Tuple[Tensor, Tensor, Tensor]:
    """calculate fps, tps, thresholds

    :param y_true: shape[N]
    :param y_score: shape[N]
    :return: Tuple[fps, tps, thresholds]
        fps: shape[T]. increase. T: n_thresholds
        tps: shape[T]. increase
        thresholds: shape[T]. decrease
    """
    device = y_score.device
    # Sort the scores in descending order
    desc_score_idxs = torch.argsort(y_score, descending=True)
    y_score = y_score[desc_score_idxs]
    y_true = y_true[desc_score_idxs]
    # threshold_idx is the index of the score jumping
    ys_diff = torch.diff(y_score, append=torch.tensor([y_score[-1] - 1], device=device))  # shape[N]
    threshold_idxs = torch.where(ys_diff)[0]  # shape[T]
    #
    tps = torch.cumsum(y_true, dim=0)[threshold_idxs]
    # or:
    # fps = torch.cumsum(1 - y_true, dim=0)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # faster

    return fps, tps, y_score[threshold_idxs]


def precision_recall_curve(
        y_true, probas_pred) -> Tuple[Tensor, Tensor, Tensor]:
    """In the PR diagram: y axis is P and x axis is R

    :param y_true: shape[N]
    :param probas_pred: shape[N]
    :return: Tuple[precision, recall, thresholds]
        precision: shape[T+1]. trend increase. T: n_thresholds
        recall: shape[T+1]. decrease
        thresholds: shape[T]. increase
    """
    dtype = torch.float32
    y_true = torch.as_tensor(y_true, dtype=dtype)
    probas_pred = torch.as_tensor(probas_pred, dtype=dtype)
    device = y_true.device
    #
    fp, tp, thresholds = _binary_clf_curve(y_true, probas_pred)  # shape[T] [T] [T]
    # tp[-1] = m+, fp[-1] = m-. Because the lowest threshold contains all true_pos, true_neg
    precision = tp / (tp + fp)
    recall = tp / tp[-1]
    # the last recall that does not promote/change omit. Because it doesn't contribute to AP
    last_ind = int(torch.searchsorted(tp, tp[-1])) + 1
    # Add PR diagram point(0, 1). and reverse for decreasing recall(same as sklearn)
    one = torch.tensor([1.], device=device)
    zero = torch.tensor([0.], device=device)
    # reverse and cat
    precision = torch.cat([torch.flip(precision[:last_ind], (0,)), one])
    recall = torch.cat([torch.flip(recall[:last_ind], (0,)), zero])
    thresholds = torch.flip(thresholds[:last_ind], (0,))
    return precision, recall, thresholds
