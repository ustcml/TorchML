# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

from dev.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix as _confusion_matrix

# In[0]
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
print(_confusion_matrix(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
"""Out[0]
[[2 0 0]
 [0 0 1]
 [1 0 2]]
tensor([[2, 0, 0],
        [0, 0, 1],
        [1, 0, 2]])
"""

# In[1]: test normalize
print()
print(_confusion_matrix(y_true, y_pred, normalize='true'))
print(confusion_matrix(y_true, y_pred, normalize='true'))

"""Out[1]
[[1.         0.         0.        ]
 [0.         0.         1.        ]
 [0.33333333 0.         0.66666667]]
tensor([[1.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 1.0000],
        [0.3333, 0.0000, 0.6667]])
"""

# In[2]: 与sklearn的不同点; test nan
y_true = [3]
y_pred = [0]
print(_confusion_matrix(y_true, y_pred, normalize='true'))
print(confusion_matrix(y_true, y_pred, normalize='true'))
"""
[[0. 0.]
 [1. 0.]]
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [1., 0., 0., 0.]])
"""
