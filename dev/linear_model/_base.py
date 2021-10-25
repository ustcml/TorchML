from abc import ABCMeta, abstractmethod
from collections import defaultdict

import torch
from typing import Tuple

from sklearn.base import BaseEstimator
from torch import Tensor
from torch.linalg import svd
from ..base import ClassifierMixin


def _solve_svd(X: Tensor, y: Tensor, alpha) -> Tensor:
    """

    :param X: shape[N, F]
    :param y: shape[N, Out]
    :param alpha: float
    :return: shape[F, Out]
    """
    # [N, Min] [Min] [Min, F]
    U, s, Vt = svd(X, full_matrices=False)
    d = s / (s ** 2 + alpha)  # [Min]
    return Vt.T @ (d[:, None] * U.T) @ y


def _data_center(X: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """数据居中. not copy

    :param X: shape[N, F]
    :param y: shape[N, Out]
    :return: Tuple[X, y, X_mean, y_mean]
        X: shape[N, F]
        y: shape[N, Out]
        X_mean: shape[F]
        y_mean: shape[Out]
    """
    X_mean = torch.mean(X, dim=0)
    y_mean = torch.mean(y, dim=0)
    X, y = X - X_mean, y - y_mean
    return X, y, X_mean, y_mean


class LinearModel(BaseEstimator, metaclass=ABCMeta):
    """Base class for Linear Models"""

    @abstractmethod
    def fit(self, X, y):
        """

        :param X: shape[N, F]
        :param y: shape[N, Out]
        :return:
        """

    def _decision_function(self, X):
        return X @ self.coef_.T + self.intercept_

    def predict(self, X) -> Tensor:
        """

        :param X: shape[N, F]
        :return: shape[N, Out]
        """
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        return self._decision_function(X)

    _data_center = staticmethod(_data_center)


class LinearClassifierMixin(ClassifierMixin):
    """Mixin for linear classifiers"""

    def decision_function(self, X):
        """

        :param X: shape[N, F]
        :return: shape[N, Out]
        """
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        return X @ self.coef_.T + self.intercept_

    def predict(self, X):
        scores = self.decision_function(X)  # 不过sigmoid/softmax
        if scores.shape[1] == 1:
            indices = (scores > 0).to(dtype=torch.long)
        else:
            indices = scores.argmax(dim=1)
        return self.classes_[indices]
