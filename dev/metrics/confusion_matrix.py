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
    dtype = torch.long
    y_true = torch.as_tensor(y_true, dtype=dtype)
    y_pred = torch.as_tensor(y_pred, dtype=dtype)
    #
    n_classes = torch.max(y_true).item() + 1
    # calculate cm
    yt_yp_idxs = y_true * n_classes + y_pred  # indices of cm.flat. cm[y_true, y_pred]
    cm = torch.bincount(yt_yp_idxs, minlength=n_classes * n_classes)
    cm = cm.reshape(n_classes, n_classes)
    # for normalize. None does no processing
    if normalize == "true":  # The sum of each true tag (row) =1
        cm = cm / torch.sum(cm, dim=1, keepdim=True)
    elif normalize == "pred":
        cm = cm / torch.sum(cm, dim=0, keepdim=True)
    elif normalize == "all":
        cm = cm / cm.sum()
    # Handles nans caused by division by zero
    cm = torch.nan_to_num(cm)
    return cm
