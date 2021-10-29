# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

from dev.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler as _MinMaxScaler
import torch

# In[0]
X = np.random.rand(100, 10)
X0 = X.copy()
s = MinMaxScaler().fit(X)
s2 = _MinMaxScaler().fit(X)
print(np.allclose(s.scale_, s2.scale_))
print(np.allclose(s.min_, s2.min_))
print(np.allclose(s.transform(X), s2.transform(X), rtol=1e-6, atol=1e-6))
print(np.all(X0 == X))  # const
print()
"""Out[0]
True
True
True
True
"""

# In[1]: test gpu

s3 = MinMaxScaler(dtype=torch.float64, device='cuda').fit(X)
print(np.allclose(s3.transform(X).cpu(), s2.transform(X)))
print(s3.transform(X).dtype, s3.transform(X).device)
print()
"""Out[1]
True
torch.float64 cuda:0
"""
