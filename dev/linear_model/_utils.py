# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 
from torch.linalg import svd
from torch import Tensor

__all__ = ["_solve_svd"]


def _solve_svd(X: Tensor, y: Tensor, alpha: float) -> Tensor:
    """

    :param X: shape[N, F]
    :param y: shape[N, Out]
    :param alpha:
    :return: shape[F, Out]
    """
    # shape[N, Min], [Min], [Min, F]
    U, s, Vt = svd(X, full_matrices=False)
    d = s / (s ** 2 + alpha)  # shape[Min]
    return Vt.T * d @ (U.T @ y)  # why bracket? N always >> F
