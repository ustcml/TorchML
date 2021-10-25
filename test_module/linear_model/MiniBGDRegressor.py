# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

import numpy as np
from dev.linear_model import MiniBGDRegressor, LinearRegression
import torch
import time


def test_module_1():
    rng = np.random.default_rng(42)
    X = rng.random((2000, 500))
    w = rng.random((500, 50))
    b = rng.random(50)
    y = X @ w + b
    #
    reg = MiniBGDRegressor(max_iter=200, momentum=0.9, device='cuda', random_state=42)
    t = time.time()
    reg.fit(X, y)
    print(reg.score(X, y))
    print(time.time() - t)
    exit(0)
    print(reg.intercept_, reg.coef_)
    """
    0.9656788204045962
    3.729032278060913
    """
    #
    reg = LinearRegression()
    reg.fit(X, y)
    print(reg.score(X, y))
    print(reg.intercept_, reg.coef_)


if __name__ == '__main__':
    test_module_1()
