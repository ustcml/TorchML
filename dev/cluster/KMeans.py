# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 
import torch
from ..utils import atleast_2d, _data_center
from torch import Tensor
from ..base import TransformerMixin
from torch import Generator

__all__ = ["KMeans"]


class KMeans(TransformerMixin):

    def __init__(self, n_clusters=8, *,
                 n_init: int = 10, max_iter: int = 300, random_state=None,
                 dtype=None, device=None):
        """init='random', algorithm='full'. 以后再优化

        :param n_clusters:
        :param n_init: 使用不同质心种子运行的次数
        """
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.dtype = dtype
        self.device = device
        #
        self.cluster_centers_ = None  # shape[N_CL, F]
        self.labels_ = None  # shape[N]
        self.inertia_ = None  # float

    def _random_init_centers(self, X):
        """

        :param X: shape[N, F]
        :return: shape[N_CL, F]
        """
        random_state = self.random_state  # type: Generator
        n_clusters = self.n_clusters
        device = self.device
        #
        seeds = torch.randperm(X.shape[0], generator=random_state, device=device)[:n_clusters]
        centers = X[seeds]
        return centers

    def _kmeans_single(self, X):
        """

        :param X:
        :return: 是否收敛
        """
        max_iter = self.max_iter
        n_clusters = self.n_clusters
        # 初始化质心
        centers = self._random_init_centers(X)  # shape[N_CL, F]
        prev_centers = centers
        for _ in range(max_iter):
            # shape[K, N, F]. 更新labels
            d2 = torch.sum((X - centers[:, None]) ** 2, dim=-1)
            labels = torch.argmin(d2, 0)
            # 更新质心
            centers = []  # 内含: shape[F]
            for i in range(n_clusters):
                centers.append(torch.mean(X[labels == i], 0))
            centers = torch.stack(centers, dim=0)  # shape[N_CL, F]
            # 质心位置不再变化，迭代停止，聚类完成
            if torch.all(centers == prev_centers):
                break
            prev_centers = centers
        # 计算惯性
        inertia = 0
        for i in range(n_clusters):
            inertia += torch.sum((X[labels == i] - centers[i]) ** 2)
        return centers, labels, inertia

    def fit(self, X):
        """

        :param X: shape[N, F]
        :return:
        """
        dtype = self.dtype = self.dtype or torch.float32
        device = self.device = self.device or (X.device if isinstance(X, Tensor) else 'cpu')
        self.random_state = self.random_state if isinstance(self.random_state, Generator) \
            else Generator(device).manual_seed(self.random_state)
        n_init = self.n_init
        #
        X = torch.as_tensor(X, dtype=dtype, device=device)
        X = atleast_2d(X)
        #
        X, X_mean = _data_center(X)  # center
        # KMeans
        best_inertia = None
        for _ in range(n_init):
            centers, labels, inertia = self._kmeans_single(X)
            #
            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centers = centers
        best_centers += X_mean
        #
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia

        return self

    def predict(self, X):
        """

        :param X:
        :return:
        """
        dtype = self.dtype
        device = self.device
        X = torch.as_tensor(X, dtype=dtype, device=device)
        centers = self.cluster_centers_
        #
        d2 = torch.sum((X - centers[:, None]) ** 2, dim=-1)
        labels = torch.argmin(d2, 0)
        return labels
