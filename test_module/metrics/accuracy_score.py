# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 
from dev.metrics import accuracy_score

# In[0]: 测试normalize
y = [1, 2, 3, 4, 5]
y_pred = [5, 4, 3, 2, 1.]
print(accuracy_score(y, y_pred))  # 0.2
print(accuracy_score(y, y_pred, normalize=False))  # 1

# In[1]: 测试bool
y = [True, False, True]
y_pred = [1, 0, 1]
print(accuracy_score(y, y_pred))  # 1.0
print(accuracy_score(y, y_pred, normalize=False))  # 3
