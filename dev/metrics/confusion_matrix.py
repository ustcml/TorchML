# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:
import torch
from torch import Tensor


def confusion_matrix(
        y_true, y_pred, *,
        normalize: str = None
) -> Tensor:
    """

    :param y_true: shape[N]
    :param y_pred: shape[N]
    :param normalize: {'true', 'pred', 'all', None}.
    :return:
    """
    y_true = torch.as_tensor(y_true)
    y_pred = torch.as_tensor(y_pred)
    #
    n_classes = torch.max(y_true).item() + 1
    # 计算cm
    yt_yp_idxs = y_true * n_classes + y_pred  # 混淆矩阵.flat的索引
    cm = torch.bincount(yt_yp_idxs, minlength=n_classes * n_classes)
    cm = cm.reshape(n_classes, n_classes)
    # 处理normalize. None 不做处理
    if normalize == "true":  # 每个true标签(row)和=1
        cm = cm / torch.sum(cm, dim=1, keepdim=True)
    elif normalize == "pred":
        cm = cm / torch.sum(cm, dim=0, keepdim=True)
    elif normalize == "all":
        cm = cm / cm.sum()
    # 处理除0导致的nan
    cm = torch.nan_to_num(cm)
    return cm
