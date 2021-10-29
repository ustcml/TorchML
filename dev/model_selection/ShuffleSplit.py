# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:
import torch
from torch import Tensor
from math import ceil


class ShuffleSplit:
    def __init__(self, n_splits: int = 10, *,
                 test_size: float = 0.1, random_state=None):
        """省去了train_size"""
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y=None):
        #
        N = len(X)
        device = X.device if isinstance(X, Tensor) else 'cpu'
        #
        n_splits = self.n_splits
        test_size = int(ceil(self.test_size * N))
        random_state = torch.Generator(device=device).manual_seed(self.random_state)
        #
        for _ in range(n_splits):
            idxs = torch.randperm(N, device=device, generator=random_state)
            train_idxs = idxs[test_size:]
            test_idxs = idxs[:test_size]
            yield train_idxs, test_idxs
