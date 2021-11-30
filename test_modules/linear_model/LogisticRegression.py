# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

import numpy as np
from dev.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression as _LogisticRegression
import time

# In[0]
rng = np.random.default_rng(42)
X = rng.standard_normal((2000, 500))
w = rng.standard_normal((500, 1))
b = rng.standard_normal(1)
y = (X @ w + b > 0).astype(np.int64)[:, 0]
#
reg = LogisticRegression(max_iter=20, device='cuda', random_state=42)
t = time.time()
reg.fit(X, y)
print(reg.score(X, y).cpu().numpy())
print(time.time() - t)
print()
"""Out[0]
0.99950004
3.529978036880493
"""

# In[1]: Compare to sklearn
reg = LogisticRegression(max_iter=20, adam=True, device='cuda', random_state=42)
reg2 = _LogisticRegression(max_iter=200, random_state=42)
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
"""Out[1].
1.0
1.463653326034546

1.0
0.05186057090759277
"""

# In[2]: multi classification
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
reg = LogisticRegression(max_iter=1000, eta0=1e-3, batch_size=32, adam=True, device='cuda', random_state=42)
reg2 = _LogisticRegression(max_iter=200, random_state=42)
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

"""Out[2]
0.9666667
5.393450975418091

0.9733333333333334
0.021935224533081055
"""
