# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:
from ._base import LinearModel
from ..base import RegressorMixin
from ..utils import atleast_2d
import torch
from torch import Tensor
import torch.nn as nn
from ..metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD


class MiniBGDRegressor(LinearModel, RegressorMixin):
    def __init__(
            self, *, alpha=0.0001,
            max_iter=1000, batch_size=64,
            eta0=0.01, momentum=0, shuffle=False, num_workers=0,
            random_state=None, dtype=None, device=None):
        """loss='squared_error', penalty='l2'

        :param alpha: 正则系数. (若为'l2'正则，则等价于weight_decay)
        :param max_iter: 最大迭代次数(epoch)
        :param eta0: 初始学习率
        :param momentum: SGD的动量
        :param shuffle: DataLoader的shuffle. 每个epoch随机混洗
        :param num_workers: DataLoader的num_workers
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.eta0 = eta0
        self.momentum = momentum
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

        :param X: shape[N, F] or [N}
        :param y: shape[N, Out] or [N]
        :return:
        """
        dtype = self.dtype = self.dtype or torch.float32
        device = self.device = self.device or (X.device if isinstance(X, Tensor) else 'cpu')
        #
        max_iter = self.max_iter
        batch_size = self.batch_size
        eta0 = self.eta0
        momentum = self.momentum
        alpha = self.alpha
        shuffle = self.shuffle
        num_workers = self.num_workers
        random_state = self.random_state
        pin_memory = False if device.lower() == "cpu" else True

        #
        X = torch.as_tensor(X, dtype=dtype)
        y = torch.as_tensor(y, dtype=dtype)
        X, y = atleast_2d(X, y)
        X, y, X_mean, y_mean = self._data_center(X, y)  # center
        X_mean, y_mean = X_mean.to(device), y_mean.to(device)
        #
        if random_state:
            torch.manual_seed(random_state)
        linear = nn.Linear(X.shape[1], y.shape[1], bias=False).to(device)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size, shuffle, pin_memory=pin_memory, num_workers=num_workers)
        optim = SGD(linear.parameters(), eta0, momentum, weight_decay=alpha)
        #
        for i in range(max_iter):
            for Xi, yi in loader:
                Xi, yi = Xi.to(device), yi.to(device)  # 显存占用更少
                y_pred = linear(Xi)
                loss = mean_squared_error(yi, y_pred)
                optim.zero_grad()
                loss.backward()
                optim.step()
        #
        self.coef_ = linear.weight.detach()
        self.intercept_ = y_mean - X_mean @ self.coef_.T
        return self
