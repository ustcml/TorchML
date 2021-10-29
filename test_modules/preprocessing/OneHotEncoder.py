# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

from dev.preprocessing import OrdinalEncoder, OneHotEncoder

from sklearn.preprocessing import OneHotEncoder as _OneHotEncoder
import pandas as pd
import numpy as np
import torch

# In[0]: categories_不同. 我认为nan是没用的，所以可以去掉
X = pd.DataFrame({
    "c": ["C", "A", "B", "C", "A", "B", np.nan],
    "s": ["a", "b", np.nan, "a", "b", "c", "c"]
})
e = _OneHotEncoder(sparse=False)
e1 = OrdinalEncoder()
e2 = OneHotEncoder()
print(e.fit_transform(X))
X = e1.fit_transform(X)
X[torch.isnan(X)] = torch.nanmedian(X, dim=0)[0]
print(e2.fit_transform(X))

"""Out[0]
[[0. 0. 1. 0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0. 1. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 1.]
 [0. 0. 1. 0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0. 1. 0. 0.]
 [0. 1. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 1. 0. 0. 1. 0.]]
tensor([[0., 0., 1., 1., 0., 0.],
        [1., 0., 0., 0., 1., 0.],
        [0., 1., 0., 0., 1., 0.],
        [0., 0., 1., 1., 0., 0.],
        [1., 0., 0., 0., 1., 0.],
        [0., 1., 0., 0., 0., 1.],
        [0., 1., 0., 0., 0., 1.]])
"""
print()

# In[1]: test gpu
X = pd.DataFrame({
    "c": ["C", "A", "B", "C", "A", "B", np.nan],
    "s": ["a", "b", np.nan, "a", "b", "c", "c"]
})
e1 = OrdinalEncoder(device='cuda')
e2 = OneHotEncoder(device='cuda')
print(e.fit_transform(X))
X = e1.fit_transform(X)
X[torch.isnan(X)] = torch.nanmedian(X, dim=0)[0]
print(e2.fit_transform(X))
"""Out[1]
tensor([[0., 0., 1., 1., 0., 0.],
        [1., 0., 0., 0., 1., 0.],
        [0., 1., 0., 0., 1., 0.],
        [0., 0., 1., 1., 0., 0.],
        [1., 0., 0., 0., 1., 0.],
        [0., 1., 0., 0., 0., 1.],
        [0., 1., 0., 0., 0., 1.]], device='cuda:0')
"""
