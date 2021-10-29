# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

from ..base import TransformerMixin
import torch
from torch import Tensor


class StandardScaler(TransformerMixin):
    def __init__(self, *, with_mean: bool = True, with_std: bool = True,
                 dtype=None, device=None):
        self.with_mean = with_mean
        self.with_std = with_std
        self.dtype = dtype
        self.device = device
        #
        self.mean_ = None  # shape[F]
        self.scale_ = None  # shape[F]

    def fit(self, X, y=None):
        """

        :param X: shape[N, F]
        :return:
        """
        dtype = self.dtype = self.dtype or torch.float32
        device = self.device = self.device or (X.device if isinstance(X, Tensor) else 'cpu')
        X = torch.as_tensor(X, dtype=dtype, device=device)
        #
        with_mean = self.with_mean
        with_std = self.with_std
        #
        if with_mean:
            self.mean_ = torch.mean(X, dim=0)
        if with_std:
            self.scale_ = torch.std(X, dim=0, unbiased=False)
        return self

    def transform(self, X) -> Tensor:
        """

        :param X: shape[N, F]
        :return: shape[N, F]
        """
        dtype = self.dtype
        device = self.device
        #
        X = torch.as_tensor(X, dtype=dtype, device=device)
        #
        if self.with_mean:
            X = X - self.mean_
        if self.with_std:
            X = X / self.scale_
        return X
