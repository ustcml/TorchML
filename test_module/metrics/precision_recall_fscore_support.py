# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

from dev.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_fscore_support as _precision_recall_fscore_support
import numpy as np

# In[0]: 二分类
y_true = np.array([0, 1, 0, 1, 1])
y_pred = np.array([0, 0, 1, 1, 1])
print(precision_recall_fscore_support(y_true, y_pred, beta=0.5))
print(precision_recall_fscore_support(y_true, y_pred, beta=0.5, average='micro'))
print(precision_recall_fscore_support(y_true, y_pred, beta=0.5, average='binary'))
print(precision_recall_fscore_support(y_true, y_pred, beta=0.5, average='macro'))
print()
print(_precision_recall_fscore_support(y_true, y_pred, beta=0.5))
print(_precision_recall_fscore_support(y_true, y_pred, beta=0.5, average='micro'))
print(_precision_recall_fscore_support(y_true, y_pred, beta=0.5, average='binary'))
print(_precision_recall_fscore_support(y_true, y_pred, beta=0.5, average='macro'))
"""Out[0]: float32精度问题忽视. 因为python的float是64位的
(tensor([0.5000, 0.6667]), tensor([0.5000, 0.6667]), tensor([0.5000, 0.6667]), tensor([2, 3]))
(0.6000000238418579, 0.6000000238418579, 0.6000000238418579, None)
(0.6666666865348816, 0.6666666865348816, 0.6666666865348816, None)
(0.5833333730697632, 0.5833333730697632, 0.5833333730697632, None)

(array([0.5       , 0.66666667]), array([0.5       , 0.66666667]), array([0.5       , 0.66666667]), array([2, 3], dtype=int64))
(0.6, 0.6, 0.6, None)
(0.6666666666666666, 0.6666666666666666, 0.6666666666666666, None)
(0.5833333333333333, 0.5833333333333333, 0.5833333333333333, None)
"""
print()

# In[1]: 多分类
y_true = np.array([0, 1, 2, 0, 1, 2])
y_pred = np.array([0, 2, 1, 0, 0, 1])
print(precision_recall_fscore_support(y_true, y_pred, beta=2))
print(precision_recall_fscore_support(y_true, y_pred, beta=2, average='micro'))
print(precision_recall_fscore_support(y_true, y_pred, beta=2, average='macro'))
print()
print(_precision_recall_fscore_support(y_true, y_pred, beta=2))
print(_precision_recall_fscore_support(y_true, y_pred, beta=2, average='micro'))
print(_precision_recall_fscore_support(y_true, y_pred, beta=2, average='macro'))
"""Out[1]
(tensor([0.6667, 0.0000, 0.0000]), tensor([1., 0., 0.]), tensor([0.9091, 0.0000, 0.0000]), tensor([2, 2, 2]))
(0.3333333432674408, 0.3333333432674408, 0.3333333432674408, None)
(0.2222222238779068, 0.3333333432674408, 0.3030303120613098, None)

(array([0.66666667, 0.        , 0.        ]), array([1., 0., 0.]), array([0.90909091, 0.        , 0.        ]), array([2, 2, 2], dtype=int64))
(0.3333333333333333, 0.3333333333333333, 0.3333333333333333, None)
(0.2222222222222222, 0.3333333333333333, 0.30303030303030304, None)
"""
print()

# In[2]: 测试nan
y_true = np.array([2])
y_pred = np.array([0])
print(precision_recall_fscore_support(y_true, y_pred, beta=0))
print(precision_recall_fscore_support(y_true, y_pred, beta=0, average='micro'))
print(precision_recall_fscore_support(y_true, y_pred, beta=0, average='macro'))
print()
print(_precision_recall_fscore_support(y_true, y_pred, beta=0, zero_division=0))
print(_precision_recall_fscore_support(y_true, y_pred, beta=0, average='micro', zero_division=0))
print(_precision_recall_fscore_support(y_true, y_pred, beta=0, average='macro', zero_division=0))
"""Out[2]
(tensor([0., 0., 0.]), tensor([0., 0., 0.]), tensor([0., 0., 0.]), tensor([0, 0, 1]))
(0.0, 0.0, 0.0, None)
(0.0, 0.0, 0.0, None)

(array([0., 0.]), array([0., 0.]), array([0., 0.]), array([0., 1.]))
(0.0, 0.0, 0.0, None)
(0.0, 0.0, 0.0, None)
"""
