# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:
from ._base import LinearModel, _solve_svd
from ..base import RegressorMixin
import torch
from torch.linalg import lstsq
from ..utils import atleast_2d
from torch import Tensor


class LinearRegression(LinearModel, RegressorMixin):
    def __init__(self, *, dtype=None, device=None):
        self.dtype = dtype
        self.device = device
        # 此处shape可能与sklearn实现不同. 我认为这样实现更清晰.
        self.coef_ = None  # shape[F, Out]
        self.intercept_ = None  # shape[Out]

    def fit(self, X, y):
        """

        :param X: shape[N, F] or [N]
        :param y: shape[N, Out] or [N]
        :return:
        """
        dtype = self.dtype = self.dtype or torch.float32
        device = self.device = self.device or (X.device if isinstance(X, Tensor) else 'cpu')
        #
        X = torch.as_tensor(X, dtype=dtype, device=device)
        y = torch.as_tensor(y, dtype=dtype, device=device)
        X, y = atleast_2d(X, y)
        # shape[N, F], shape[N, Out], shape[F], shape[Out]
        X, y, X_mean, y_mean = self._data_center(X, y)  # center

        # or:
        # self.coef_ = _solve_svd(X, y, 0).T
        self.coef_ = lstsq(X, y)[0].T  # 更快
        self.intercept_ = y_mean - X_mean @ self.coef_.T

        return self
