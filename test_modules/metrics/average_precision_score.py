# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import numpy as np
from dev.metrics import average_precision_score
from sklearn.metrics import average_precision_score as _average_precision_score
import torch

# In[0]
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
print(average_precision_score(y_true, y_scores))
print(_average_precision_score(y_true, y_scores))
print()
"""Out[0]
tensor(0.8333)
0.8333333333333333
"""

# In[1]: test gpu
y_true = torch.tensor([0, 0, 1, 1], device='cuda')
y_scores = torch.tensor([0.1, 0.4, 0.35, 0.8], device='cuda')
print(average_precision_score(y_true, y_scores))
print()
"""Out[1]
tensor(0.8333, device='cuda:0')
"""
