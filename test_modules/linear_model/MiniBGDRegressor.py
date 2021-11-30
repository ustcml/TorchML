# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

import numpy as np
from dev.linear_model import MiniBGDRegressor, LinearRegression
from sklearn.linear_model import SGDRegressor
import time

# In[0]
rng = np.random.default_rng(42)
X = rng.random((2000, 500))
w = rng.random((500, 1))
b = rng.random(1)
y = X @ w + b + rng.normal(scale=0.1, size=(2000, 1))
#
reg = MiniBGDRegressor(max_iter=20, device='cuda', random_state=42)
t = time.time()
reg.fit(X, y)
print(reg.score(X, y).cpu().numpy())
print(time.time() - t)
print()
"""Out[0]
0.99945986
2.948368549346924
"""

# In[1]: Compare with linear regression
reg = LinearRegression(device='cuda')
t = time.time()
reg.fit(X, y)
print(reg.score(X, y).cpu().numpy())
print(time.time() - t)
print()
"""Out[1].
0.99949783
0.14960050582885742
"""

# In[2]: Compare to SGD(sklearn), which is written in Cython
rng = np.random.default_rng(42)
X = rng.random((2000, 500))
w = rng.random(500)
b = rng.random()
y = X @ w + b + rng.normal(scale=0.1, size=(2000))
#
reg = MiniBGDRegressor(max_iter=20, adam=True, batch_size=64, device='cuda', random_state=42)
reg2 = SGDRegressor(max_iter=200, random_state=42)
t = time.time()
reg.fit(X, y)
print(reg.score(X, y).cpu().numpy())
print(time.time() - t)
print()
#
t = time.time()
reg2.fit(X, y)
print(reg2.score(X, y))
print(time.time() - t)
print()
"""Out[2].
0.99945986
0.7360303401947021

0.9984602848920857
0.09674191474914551
"""
