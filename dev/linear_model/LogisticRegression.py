# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:
from ..base import ClassifierMixin
from ._base import LinearClassifierMixin
from ..utils import atleast_2d
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD, Adam


class LogisticRegression(LinearClassifierMixin, ClassifierMixin):
    def __init__(
            self, *, alpha=0.0001,
            max_iter=1000, batch_size=32,
            eta0=0.01, adam=False, shuffle=True, num_workers=0,
            random_state=None, dtype=None, device=None):
        """penalty='l2'

        :param alpha: Regularization coefficient. =weight_decay
        :param max_iter: maximum number of iterations(epoch)
        :param eta0: Initial learning rate
        :param adam:
            False: SGD: momentum=0.9, nesterov=True
            True: Adam: beta=(0.9, 0.999), eps=1e-8
        :param shuffle: for DataLoader. every epoch shuffle
        :param num_workers: for DataLoader
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.eta0 = eta0 or (1e-3 if adam else 1e-2)
        self.adam = adam
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.random_state = random_state
        self.dtype = dtype
        self.device = device
        #
        self.coef_ = None  # shape[F, Out]
        self.intercept_ = None  # shape[Out]

    def fit(self, X, y):
        """

        :param X: shape[N, F] or [N]
        :param y: shape[N]
        :return:
        """
        dtype = self.dtype = self.dtype or torch.float32
        device = self.device = self.device or (X.device if isinstance(X, Tensor) else 'cpu')
        #
        max_iter = self.max_iter
        batch_size = self.batch_size
        eta0 = self.eta0
        adam = self.adam
        alpha = self.alpha
        shuffle = self.shuffle
        num_workers = self.num_workers
        random_state = self.random_state
        pin_memory = False if device.lower() == "cpu" else True
        #
        X = torch.as_tensor(X, dtype=dtype)
        y = torch.as_tensor(y)
        X = atleast_2d(X)
        #
        num_classes = torch.max(y) + 1
        if num_classes > 2:  # multi classification. y: shape[N, C]
            y = F.one_hot(y.long(), -1)
        else:  # binary classification. y: shape[N, 1]
            y = y[:, None]
        y = y.to(dtype=dtype)
        #
        if random_state:
            torch.manual_seed(random_state)
        linear = nn.Linear(X.shape[1], y.shape[1], bias=True).to(device)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size, shuffle, pin_memory=pin_memory, num_workers=num_workers)
        optim = SGD(linear.parameters(), eta0, 0.9, weight_decay=alpha, nesterov=True) if not adam else \
            Adam(linear.parameters(), eta0, (0.9, 0.999), eps=1e-8, weight_decay=alpha)
        #
        for i in range(max_iter):
            for Xi, yi in loader:
                Xi, yi = Xi.to(device), yi.to(device)  # less memory
                y_pred = linear(Xi)
                if y.shape[1] == 1:
                    loss = F.binary_cross_entropy_with_logits(y_pred, yi)
                else:
                    loss = F.cross_entropy(y_pred, yi)
                optim.zero_grad()
                loss.backward()
                optim.step()
        #
        self.coef_ = linear.weight.detach()
        self.intercept_ = linear.bias.detach()
        return self
