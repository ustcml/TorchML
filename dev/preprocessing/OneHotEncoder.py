# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import torch
from torch import Tensor
from ..base import TransformerMixin


class OneHotEncoder(TransformerMixin):
    """此处与sklearn实现不同. 只能处理number(e.g. int/long)的情况"""

    def __init__(self, *, dtype=None, device=None):
        """sparse=False"""
        self.dtype = dtype
        self.device = device

    def fit(self, X=None, y=None):
        self.dtype = self.dtype or torch.float32
        self.device = self.device or 'cpu'
        return self

    def transform(self, X) -> Tensor:
        """数据中不能有nan

        :param X: shape[N, F]
        :return: shape[N, X]
        """
        dtype = self.dtype
        device = self.device
        # test nan
        if torch.any(torch.isnan(X)):
            raise ValueError("X has nan")
        #
        X = torch.as_tensor(X, dtype=torch.long, device=device)
        res = []  # 列优先
        for Xi in X.T:
            Xi_classes = torch.max(Xi).item() + 1
            res.append(torch.eye(Xi_classes, dtype=dtype, device=device)[Xi])
        return torch.cat(res, dim=1)

    def fit_transform(self, X, y=None, **fit_params) -> Tensor:
        # test nan
        if torch.any(torch.isnan(X)):
            raise ValueError("X has nan")
        return self.fit(X).transform(X)
