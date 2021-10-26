from abc import ABCMeta, abstractmethod

import torch

from sklearn.base import BaseEstimator
from torch import Tensor
from torch.linalg import svd
from ..base import ClassifierMixin
from ..utils import _data_center

__all__ = ["_solve_svd", "LinearModel", "LinearClassifierMixin"]


def _solve_svd(X: Tensor, y: Tensor, alpha: float) -> Tensor:
    """

    :param X: shape[N, F]
    :param y: shape[N, Out]
    :param alpha:
    :return: shape[F, Out]
    """
    # shape[N, Min], [Min], [Min, F]
    U, s, Vt = svd(X, full_matrices=False)
    d = s / (s ** 2 + alpha)  # shape[Min]
    return Vt.T @ (d[:, None] * U.T) @ y


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
        scores = self.decision_function(X)  # 不经过sigmoid/softmax
        if scores.shape[1] == 1:
            indices = (scores > 0).to(dtype=torch.long)
        else:
            indices = scores.argmax(dim=1)
        return self.classes_[indices]
