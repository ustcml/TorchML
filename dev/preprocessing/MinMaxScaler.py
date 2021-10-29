# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

from ..base import TransformerMixin
import torch
from torch import Tensor
from typing import Tuple


class MinMaxScaler(TransformerMixin):
    def __init__(self, feature_range: Tuple[int] = (0, 1), *, clip: bool = False, dtype=None, device=None):
        self.feature_range = feature_range
        self.clip = clip
        self.dtype = dtype
        self.device = device
        #
        self.min_ = None  # shape[F]
        self.scale_ = None  # shape[F]
        self.data_min_ = None  # shape[F]
        self.data_max_ = None  # shape[F]
        self.data_range_ = None

    def fit(self, X, y=None):
        """不允许有nan. 请在缺省值填充完后再fit. (与sklearn不同)

        :param X: shape[N, F]
        :return:
        """
        dtype = self.dtype = self.dtype or torch.float32
        device = self.device = self.device or (X.device if isinstance(X, Tensor) else 'cpu')
        X = torch.as_tensor(X, dtype=dtype, device=device)
        #
        feature_range = self.feature_range

        #
        data_min = torch.min(X, dim=0)[0]
        data_max = torch.max(X, dim=0)[0]
        data_range = data_max - data_min
        #
        self.scale_ = (feature_range[1] - feature_range[0]) / data_range
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self

    def transform(self, X) -> Tensor:
        """

        :param X: shape[N, F]
        :return: shape[N, F]
        """
        dtype = self.dtype
        device = self.device
        clip = self.clip
        #
        X = torch.as_tensor(X, dtype=dtype, device=device)
        #
        X = X * self.scale_
        X = X + self.min_
        if clip:
            X = torch.clip(X, self.feature_range[0], self.feature_range[1])
        return X
