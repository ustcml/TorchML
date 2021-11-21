# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 
import torch
from ..utils import atleast_2d, _data_center, one_hot
from torch import Tensor
from ..base import TransformerMixin
from torch import Generator
from typing import Tuple
from ..metrics import euclidean_distances

__all__ = ["KMeans"]


class KMeans(TransformerMixin):

    def __init__(self, n_clusters=8, *,
                 n_init: int = 10, max_iter: int = 300, random_state=None,
                 dtype=None, device=None):
        """init='random', algorithm='full'. Optimize it later(e.g. kmeans++)

        :param n_clusters:
        :param n_init: Number of seeds to runs for different initial centers
        """
        self.n_clusters = n_clusters  # K
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.dtype = dtype
        self.device = device
        #
        self.cluster_centers_ = None  # shape[K, F]
        self.labels_ = None  # shape[N]
        self.inertia_ = None  # float

    def _random_init_centers(self, X: Tensor) -> Tensor:
        """

        :param X: shape[N, F]
        :return: shape[K, F]
        """
        random_state = self.random_state  # type: Generator
        n_clusters = self.n_clusters
        device = self.device
        #
        seeds = torch.randperm(X.shape[0], generator=random_state, device=device)[:n_clusters]
        centers = X[seeds]
        return centers

    def _update_labels(self, X: Tensor, centers: Tensor) -> Tensor:
        """

        :param X: shape[N, F]
        :param centers: shape[K, F]
        :return: shape[N]
        """
        d2 = euclidean_distances(X, centers, squared=True)  # shape[N, K]
        labels = torch.argmin(d2, 1)  # shape[N]
        return labels

    def _update_centers(self, X: Tensor, labels: Tensor) -> Tensor:
        """Ot(KNF).

        :param X: shape[N, F]
        :param labels: shape[N]
        :return: shape[K, F]
        """
        dtype = X.dtype
        #
        mask = one_hot(labels, dtype=dtype)  # shape[N, K]
        n_active = torch.sum(mask, dim=0)
        # [K, N] @ [N, F] / [K, 1]
        centers = mask.T @ X / n_active[:, None]  # Ot(KNF)
        return centers

    def _update_centers2(self, X: Tensor, labels: Tensor) -> Tensor:
        """Because of the for loop, the _update_centers() method is used.

        :param X: shape[N, F]
        :param labels: shape[N]
        :return: shape[K, F]
        """
        n_clusters = self.n_clusters
        #
        centers = []  # Len[K] of shape[F]
        for i in range(n_clusters):
            centers.append(torch.mean(X[labels == i], 0))
        centers = torch.stack(centers, dim=0)  # shape[K, F]
        return centers

    def _kmeans_single(self, X: Tensor) -> Tuple[Tensor, Tensor, float]:
        """

        :param X: shape[N, F]
        :return: Tuple[centers, labels, inertia]
            centers: shape[K, F]
            labels: shape[N]
            inertia: float
        """
        max_iter = self.max_iter
        n_clusters = self.n_clusters
        # Initialize the centers
        centers = self._random_init_centers(X)  # shape[K, F]
        prev_labels = None
        for _ in range(max_iter):
            # Update labels for samples
            labels = self._update_labels(X, centers)
            # Update centers
            centers = self._update_centers(X, labels)
            # Judge convergence
            if prev_labels is not None and torch.all(labels == prev_labels):
                break
            prev_labels = labels
        # Calculate inertia
        inertia = 0.
        for i in range(n_clusters):
            d2 = euclidean_distances(X[labels == i], centers[i], squared=True)  # [X]
            inertia += torch.sum(d2).item()
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

        :param X: shape[N, F]
        :return: shape[N]
        """
        dtype = self.dtype
        device = self.device
        X = torch.as_tensor(X, dtype=dtype, device=device)
        centers = self.cluster_centers_
        #
        d2 = euclidean_distances(X, centers, squared=True)  # shape[N, K]
        labels = torch.argmin(d2, 1)
        return labels

    def fit_predict(self, X):
        """

        :param X: shape[N, F]
        :return: shape[N]
        """
        self.fit(X)
        return self.labels_

    def transform(self, X):
        """

        :param X: shape[N, F]
        :return: shape[N, K]
        """
        dtype = self.dtype
        device = self.device
        centers = self.cluster_centers_
        #
        X = torch.as_tensor(X, dtype=dtype, device=device)
        X = atleast_2d(X)
        #
        return euclidean_distances(X, centers)
