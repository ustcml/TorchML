# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

from dev.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import StandardScaler as _StandardScaler
import torch

# In[0]
X = np.random.rand(100, 10)
X0 = X.copy()
s = StandardScaler().fit(X)
s2 = _StandardScaler().fit(X)
print(np.allclose(s.mean_, s2.mean_))
print(np.allclose(s.scale_, s2.scale_))
print(np.allclose(s.transform(X), s2.transform(X), rtol=1e-6, atol=1e-6))
print(np.all(X0 == X))
print()
"""Out[0]
True
True
True
True
"""

# In[1]: test gpu

s3 = StandardScaler(dtype=torch.float64, device='cuda').fit(X)
print(np.allclose(s3.transform(X).cpu(), s2.transform(X)))
print(s3.transform(X).dtype, s3.transform(X).device)
print()
"""Out[1]
True
torch.float64 cuda:0
"""

# In[2]: nan
X = np.array([[1, 2, np.nan]])
s = StandardScaler().fit(X)
print(s.mean_, s.scale_, s.transform(X))
"""Out[2]
tensor([1., 2., nan]) tensor([0., 0., nan]) tensor([[nan, nan, nan]])
"""