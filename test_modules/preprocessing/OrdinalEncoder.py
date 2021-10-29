# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

from dev.preprocessing import OrdinalEncoder

from sklearn.preprocessing import OrdinalEncoder as _OrdinalEncoder
import pandas as pd
import numpy as np

# In[0]: categories_不同. 我认为nan是没用的，所以可以去掉
X = pd.DataFrame({
    "c": ["C", "A", "B", "C", "A", "B", np.nan],
    "s": ["a", "b", np.nan, "a", "b", "c", "c"]
})
e = OrdinalEncoder()
e2 = _OrdinalEncoder()
e.fit(X)
e2.fit(X)
print(e.categories_, e2.categories_)
print(e.transform(X))
print(e2.transform(X))
print()
"""Out[0]
[['A', 'B', 'C'], ['a', 'b', 'c']] [array(['A', 'B', 'C', nan], dtype=object), array(['a', 'b', 'c', nan], dtype=object)]
tensor([[2., 0.],
        [0., 1.],
        [1., nan],
        [2., 0.],
        [0., 1.],
        [1., 2.],
        [nan, 2.]])
[[ 2.  0.]
 [ 0.  1.]
 [ 1. nan]
 [ 2.  0.]
 [ 0.  1.]
 [ 1.  2.]
 [nan  2.]]
"""
# In[1]: 不同. 传入List[Union[Series, List]]时，是列优先.
# In[2]: test categories
x1 = pd.Series(["C", "A", "B", "C", "A", "B", np.nan])
x2 = ["a", "b", np.nan, "a", "b", "c", "c"]
e = OrdinalEncoder(categories=[['C', 'B', 'A'], ["c", "b", "a"]])
e2 = _OrdinalEncoder()
e.fit([x1, x2])
e2.fit([x1, x2])
print(e.categories_)
print(e2.categories_)
print(e.transform([x1, x2]))
print(e2.transform([x1, x2]))
"""Out[1] Out[2]
[['C', 'B', 'A'], ['c', 'b', 'a']]
[array(['C', 'a'], dtype=object), array(['A', 'b'], dtype=object), array(['B', nan], dtype=object), array(['C', 'a'], dtype=object), array(['A', 'b'], dtype=object), array(['B', 'c'], dtype=object), array(['c', nan], dtype=object)]
tensor([[0., 2.],
        [2., 1.],
        [1., nan],
        [0., 2.],
        [2., 1.],
        [1., 0.],
        [nan, 0.]])
[[ 0.  0.  0.  0.  0.  0. nan]
 [ 1.  1. nan  1.  1.  1.  0.]]
"""
