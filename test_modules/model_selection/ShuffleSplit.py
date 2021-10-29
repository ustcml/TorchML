from sklearn.model_selection import ShuffleSplit as _ShuffleSplit
import numpy as np
from dev.model_selection import ShuffleSplit
import torch

# In[0]
ss = _ShuffleSplit(5, test_size=0.3, random_state=42)
ss2 = ShuffleSplit(5, test_size=0.3, random_state=42)
X = np.random.randn(10, 10)
y = np.random.randn(10, )
print(list(ss.split(X, y)))
print()
print(list(ss2.split(X, y)))
print()
"""Out[0]: torch和numpy随机数生成算法不同
[(array([0, 7, 2, 9, 4, 3, 6]), array([8, 1, 5])), (array([5, 3, 4, 7, 9, 6, 2]), array([0, 1, 8])), (array([6, 8, 5, 3, 7, 1, 4]), array([9, 2, 0])), (array([2, 8, 0, 3, 4, 5, 9]), array([1, 7, 6])), (array([8, 0, 7, 6, 3, 2, 9]), array([1, 5, 4]))]

[(tensor([8, 4, 5, 0, 9, 3, 7]), tensor([2, 6, 1])), (tensor([9, 4, 6, 1, 0, 7, 2]), tensor([3, 5, 8])), (tensor([3, 1, 6, 9, 8, 4, 2]), tensor([7, 5, 0])), (tensor([1, 4, 3, 7, 0, 8, 6]), tensor([9, 2, 5])), (tensor([4, 1, 7, 5, 3, 0, 8]), tensor([9, 6, 2]))]
"""

# In[1]: test gpu. gpu的随机数生成器和cpu产生的不同...很迷惑. 但这是正确的
X = torch.randn(10, 10, device='cuda')
y = torch.randn(10, device='cuda')
ss2 = ShuffleSplit(5, test_size=0.3, random_state=42)
print(list(ss2.split(X, y)))
print()
"""Out[1]
[(tensor([3, 5, 8, 4, 7, 1, 9], device='cuda:0'), tensor([6, 0, 2], device='cuda:0')), (tensor([3, 4, 1, 5, 8, 0, 9], device='cuda:0'), tensor([7, 6, 2], device='cuda:0')), (tensor([3, 9, 2, 5, 4, 1, 6], device='cuda:0'), tensor([0, 7, 8], device='cuda:0')), (tensor([2, 4, 8, 7, 6, 3, 5], device='cuda:0'), tensor([9, 0, 1], device='cuda:0')), (tensor([4, 8, 2, 1, 9, 6, 7], device='cuda:0'), tensor([0, 3, 5], device='cuda:0'))]
"""

# In[2]: 测试唯一性. 此包未发布
from dev.model_selection.np import ShuffleSplit

ss4 = ShuffleSplit(5, test_size=0.3, random_state=42)
print(list(ss4.split(X, y)))
"""Out[2]
[(array([0, 7, 2, 9, 4, 3, 6]), array([8, 1, 5])), (array([5, 3, 4, 7, 9, 6, 2]), array([0, 1, 8])), (array([6, 8, 5, 3, 7, 1, 4]), array([9, 2, 0])), (array([2, 8, 0, 3, 4, 5, 9]), array([1, 7, 6])), (array([8, 0, 7, 6, 3, 2, 9]), array([1, 5, 4]))]
"""
