# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

import numpy as np
from sklearn.linear_model import LinearRegression as _LinearRegression
import time
from dev.linear_model import LinearRegression
import torch

# In[0]: 测试运行时间
# In[1]: 测试准确性
# In[2]: 测试shape
rng = np.random.default_rng(42)
X = rng.random((2000, 500), dtype=np.float32)
y = rng.random((2000, 50), dtype=np.float32)

# sklearn warm up
t = time.time()
reg = _LinearRegression()
reg.fit(X, y)
#
y_sk = reg.predict(X)
s_sk = reg.score(X, y)
c_sk = reg.coef_
i_sk = reg.intercept_
print("time: %.6f" % (time.time() - t))

# sklearn
t = time.time()
reg = _LinearRegression()
reg.fit(X, y)
reg.predict(X)
print("time: %.6f" % (time.time() - t))

# torch warm up gpu
t = time.time()
reg = LinearRegression(device='cuda')
reg.fit(X, y)
#
y_t = reg.predict(X).cpu().numpy()
s_t = reg.score(X, y)
c_t = reg.coef_.cpu().numpy()
i_t = reg.intercept_.cpu().numpy()
print("time: %.6f" % (time.time() - t))

# torch gpu
t = time.time()
reg = LinearRegression(device='cuda')
reg.fit(X, y)
reg.predict(X)
print("time: %.6f" % (time.time() - t))

# torch cpu warm up
t = time.time()
reg = LinearRegression()
reg.fit(X, y)
reg.predict(X)
print("time: %.6f" % (time.time() - t))

# torch cpu
t = time.time()
reg = LinearRegression()
reg.fit(X, y)
reg.predict(X)
print("time: %.6f" % (time.time() - t))
"""Out[0]
time: 0.467718
time: 0.458780
time: 2.861659
time: 0.010971
time: 0.021941
time: 0.012964
"""
# 判断正确性
print(np.allclose(y_sk, y_t, rtol=1e-4, atol=1e-5))
print(np.allclose(s_sk, s_t, rtol=1e-4, atol=1e-6))
print(np.allclose(c_sk, c_t, rtol=1e-4, atol=1e-6))
print(np.allclose(i_sk, i_t, rtol=1e-4, atol=1e-5))
"""Out[1]
True
True
True
True
"""

# 测试shape
print(y_sk.shape, y_t.shape)
print(c_sk.shape, c_t.shape)
print(i_sk.shape, i_t.shape)
"""Out[2]
(20000, 50) (20000, 50)
(50, 500) (50, 500)
(50,) (50,)
"""

# In[3]: 与sklearn的区别: shape不同
rng = np.random.default_rng(42)
X = rng.random((2, 1), dtype=np.float32)
y = rng.random((2,), dtype=np.float32)

# sklearn
reg = _LinearRegression()
reg.fit(X, y)
#
y_sk = reg.predict(X)
s_sk = reg.score(X, y)
c_sk = reg.coef_
i_sk = reg.intercept_

# torch gpu
reg = LinearRegression(device='cuda')
reg.fit(X, y)
#
y_t = reg.predict(X).cpu().numpy()
s_t = reg.score(X, y)
c_t = reg.coef_.cpu().numpy()
i_t = reg.intercept_.cpu().numpy()
print(y_sk, y_t)
print(s_sk, s_t)
print(c_sk, c_t)
print(i_sk, i_t)
"""Out[3]
[0.6545714  0.43887842] [[0.6545714 ]
 [0.43887842]]
1.0 1.0
[-0.3150159] [[-0.31501594]]
0.68268687 [0.68268687]
"""
