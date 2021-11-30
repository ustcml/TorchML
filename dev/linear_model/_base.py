from abc import ABCMeta, abstractmethod

import torch
from torch import Tensor
from ..base import ClassifierMixin
from ..utils import atleast_2d

__all__ = ["LinearModel", "LinearClassifierMixin"]


class LinearModel(metaclass=ABCMeta):
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
        X = atleast_2d(X)
        return self._decision_function(X)


class LinearClassifierMixin(ClassifierMixin):
    """Mixin for linear classifiers"""

    def decision_function(self, X):
        """

        :param X: shape[N, F]
        :return: shape[N, Out]
        """
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        return X @ self.coef_.T + self.intercept_

    def predict_proba(self, X):
        """

        :param X: shape[N, F]
        :return: shape[N, Out]
        """
        scores = self.decision_function(X)  # Without sigmoid/softmax
        if scores.shape[1] == 1:
            y_pred_proba = torch.sigmoid(scores)
        else:
            y_pred_proba = torch.softmax(scores, dim=-1)
        return y_pred_proba

    def predict(self, X):
        """

        :param X: shape[N, F]
        :return: shape[N]
        """
        scores = self.decision_function(X)  # Without sigmoid/softmax
        if scores.shape[1] == 1:
            y_pred = (scores[:, 0] > 0).to(dtype=torch.long)
        else:
            y_pred = scores.argmax(dim=1)
        return y_pred
