# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import numpy as np
from dev.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_curve as _precision_recall_curve
import torch

# In[0]
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
print(precision_recall_curve(y_true, y_scores))
print(_precision_recall_curve(y_true, y_scores))
print()
"""Out[0]
(tensor([0.6667, 0.5000, 1.0000, 1.0000]), tensor([1.0000, 0.5000, 0.5000, 0.0000]), tensor([0.3500, 0.4000, 0.8000]))
(array([0.66666667, 0.5       , 1.        , 1.        ]), array([1. , 0.5, 0.5, 0. ]), array([0.35, 0.4 , 0.8 ]))
"""

# In[1]: test gpu
y_true = torch.tensor([0, 0, 1, 1], device='cuda')
y_scores = torch.tensor([0.1, 0.4, 0.35, 0.8], device='cuda')
print(precision_recall_curve(y_true, y_scores))
print()
"""Out[1]
(tensor([0.6667, 0.5000, 1.0000, 1.0000], device='cuda:0'), tensor([1.0000, 0.5000, 0.5000, 0.0000], device='cuda:0'), tensor([0.3500, 0.4000, 0.8000], device='cuda:0'))
"""
