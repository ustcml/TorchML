# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

import numpy as np
from dev.linear_model import MiniBGDRegressor, LinearRegression
import torch
import time

# In[0]
rng = np.random.default_rng(42)
X = rng.random((2000, 500))
w = rng.random((500, 50))
b = rng.random(50)
y = X @ w + b + rng.normal(scale=0.1, size=(2000, 50))
#
reg = MiniBGDRegressor(max_iter=200, momentum=0.9, device='cuda', random_state=42)
t = time.time()
reg.fit(X, y)
print(reg.score(X, y))
print(time.time() - t)
"""Out[0]
0.9902716875076294
10.177897214889526
"""
# In[1]: 与线性回归比较
reg = LinearRegression()
reg.fit(X, y)
print(reg.score(X, y))
"""Out[1].
0.9994344115257263
"""
