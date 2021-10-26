# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:
from .metrics import r2_score, accuracy_score
from .utils import atleast_2d
import torch

__all__ = ["RegressorMixin", "ClassifierMixin", "TransformerMixin"]


class RegressorMixin:
    """Mixin class for all regression estimators in scikit-learn."""

    def score(self, X, y) -> float:
        dtype = self.dtype
        device = self.device
        #
        X = torch.as_tensor(X, dtype=dtype, device=device)
        y = torch.as_tensor(y, dtype=dtype, device=device)
        y_pred = self.predict(X)
        y = atleast_2d(y)
        return r2_score(y, y_pred)


class ClassifierMixin:
    """Mixin class for all classifiers in scikit-learn."""

    def score(self, X, y):
        dtype = self.dtype
        device = self.device
        #
        X = torch.as_tensor(X, dtype=dtype, device=device)
        y = torch.as_tensor(y, dtype=dtype, device=device)
        y_pred = self.predict(X)
        y = atleast_2d(y)
        return accuracy_score(y, y_pred)


class TransformerMixin:
    """Mixin class for all transformers in scikit-learn."""

    def fit_transform(self, X, y=None, **fit_params):
        dtype = self.dtype
        device = self.device
        #
        X = torch.as_tensor(X, dtype=dtype, device=device)
        if y is None:
            return self.fit(X, **fit_params).transform(X)
        else:
            y = torch.as_tensor(y, dtype=dtype, device=device)
            return self.fit(X, y, **fit_params).transform(X)
