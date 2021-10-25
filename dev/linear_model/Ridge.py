# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:
from ._base import LinearModel, _solve_svd
from ..base import RegressorMixin
import torch
from ..utils import atleast_2d


class Ridge(LinearModel, RegressorMixin):
    def __init__(self, *, alpha=1.0, dtype=None, device=None):
        """使用svd实现"""
        self.dtype = dtype
        self.device = device
        self.alpha = alpha
        # 此处shape可能与sklearn实现不同. 我认为这样实现更清晰.
        self.coef_ = None  # shape[F, Out]
        self.intercept_ = None  # shape[Out]

    def fit(self, X, y):
        """

        :param X: shape[N, F] or [N]
        :param y: shape[N, Out] or [N]
        :return:
        """
        dtype = self.dtype = torch.float32
        device = self.device
        alpha = self.alpha
        #
        X = torch.as_tensor(X, dtype=dtype, device=device)
        y = torch.as_tensor(y, dtype=dtype, device=device)
        X, y = atleast_2d(X, y)
        # shape[N, F], shape[N, Out], shape[F], shape[Out]
        X, y, X_mean, y_mean = self._data_center(X, y)  # center

        # or:
        # self.coef_ = (pinv(X) @ y).T
        self.coef_ = _solve_svd(X, y, alpha).T
        self.intercept_ = y_mean - X_mean @ self.coef_.T

        return self
