# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:
from .metrics import r2_score, accuracy_score
import torch


class RegressorMixin:
    def score(self, X, y) -> float:
        dtype = self.dtype
        device = self.device
        #
        X = torch.as_tensor(X, dtype=dtype, device=device)
        y = torch.as_tensor(y, dtype=dtype, device=device)
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


class ClassifierMixin:
    def score(self, X, y):
        dtype = self.dtype
        device = self.device
        #
        X = torch.as_tensor(X, dtype=dtype, device=device)
        y = torch.as_tensor(y, dtype=dtype, device=device)
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
