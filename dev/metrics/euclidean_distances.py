# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import torch
from torch import Tensor


def euclidean_distances(
        X, Y=None, *,
        squared=False
) -> Tensor:
    """

    :param X: shape[N, F]
    :param Y: shape[K, F]
    :param squared:
    :return: shape[N, K]
    """
    dtype = torch.float32
    X = torch.as_tensor(X, dtype=dtype)
    Y = torch.as_tensor(Y, dtype=dtype)
    # sklearn的中的算法没看懂(可能更优)...
    distances = torch.sum((X[:, None] - Y[None]) ** 2, dim=-1)  # 第二范数^2
    return distances if squared else torch.sqrt(distances)
