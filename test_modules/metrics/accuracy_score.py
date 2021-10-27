# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 
from dev.metrics import accuracy_score
import torch

# In[0]: 测试normalize
# In[1]: 测试bool
y = [1, 2, 3, 4, 5]
y_pred = [5, 4, 3, 2, 1.]
print(accuracy_score(y, y_pred))
print(accuracy_score(y, y_pred, normalize=False))
print()
#
y = [True, False, True]
y_pred = [1, 0, 1]
print(accuracy_score(y, y_pred))
print(accuracy_score(y, y_pred, normalize=False))
print()
"""Out[0] Out[1]
tensor(0.2000)
tensor(1)

tensor(1.)
tensor(3)
"""
# In[2]: test gpu
y = torch.tensor([1, 2, 3, 4, 5], device='cuda')
y_pred = torch.tensor([5, 4, 3, 2, 1.], device='cuda')
print(accuracy_score(y, y_pred))
print(accuracy_score(y, y_pred, normalize=False))
"""Out[2]
tensor(0.2000, device='cuda:0')
tensor(1, device='cuda:0')
"""
