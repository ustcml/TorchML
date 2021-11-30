# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:
from .metrics import r2_score, accuracy_score
from .utils import atleast_2d
import torch
from torch import Tensor

__all__ = ["RegressorMixin", "ClassifierMixin", "TransformerMixin"]


class RegressorMixin:
    """Mixin class for all regression estimators"""

    def score(self, X, y) -> Tensor:
        dtype = self.dtype
        device = self.device
        #
        X = torch.as_tensor(X, dtype=dtype, device=device)
        y = torch.as_tensor(y, dtype=dtype, device=device)
        y_pred = self.predict(X)
        y = atleast_2d(y)
        return r2_score(y, y_pred)


class ClassifierMixin:
    """Mixin class for all classifiers"""

    def score(self, X, y) -> Tensor:
        dtype = self.dtype
        device = self.device
        #
        X = torch.as_tensor(X, dtype=dtype, device=device)
        y = torch.as_tensor(y, dtype=dtype, device=device)
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


class TransformerMixin:
    """Mixin class for all transformers"""

    def fit_transform(self, X, y=None, **fit_params) -> Tensor:
        dtype = self.dtype
        device = self.device
        #
        X = torch.as_tensor(X, dtype=dtype, device=device)
        if y is None:
            return self.fit(X, **fit_params).transform(X)
        else:
            y = torch.as_tensor(y, dtype=dtype, device=device)
            return self.fit(X, y, **fit_params).transform(X)
