# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 
import torch
from ..utils import atleast_2d, _data_center
from torch.linalg import svd
from torch import Tensor
from ..base import TransformerMixin

__all__ = ["PCA"]


class PCA(TransformerMixin):
    """使用full svd实现"""

    def __init__(self, n_components=None, *, dtype=None, device=None):
        self.n_components = n_components
        self.dtype = dtype
        self.device = device
        #
        self.mean_ = None  # shape[F]
        self.singular_values_ = None  # shape[N_CP]
        self.components_ = None  # shape[N_CP, F]
        self.explained_variance_ = None  # shape[N_CP]
        self.explained_variance_ratio_ = None  # shape[N_CP]

    def fit(self, X):
        """

        :param X: shape[N, F]
        :return:
        """
        dtype = self.dtype = self.dtype or torch.float32
        device = self.device = self.device or (X.device if isinstance(X, Tensor) else 'cpu')
        #
        n_components = self.n_components
        X = torch.as_tensor(X, dtype=dtype, device=device)
        X = atleast_2d(X)
        #
        X, self.mean_ = _data_center(X)  # center
        U, S, Vt = svd(X, full_matrices=False)
        #
        self.singular_values_ = S[:n_components]  # shape[N_CP]
        self.components_ = Vt[:n_components]  # shape[N_CP, F]
        #
        self.explained_variance_ = (S ** 2) / (X.shape[0] - 1)
        total_var = torch.sum(self.explained_variance_)  # for 归一化
        self.explained_variance_ = self.explained_variance_[:n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        return self

    def transform(self, X) -> Tensor:
        """

        :param X: shape[N, F]
        :return: shape[N, N_CP]
        """
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        return X @ self.components_.T

    def inverse_transform(self, X):
        """

        :param X: shape[N, N_CP]
        :return: shape[N, F]
        """
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        # self.components_为正交矩阵. 逆等于转置
        return X @ self.components_ + self.mean_
