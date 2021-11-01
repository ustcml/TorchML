# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:
from dev.cluster import KMeans
from sklearn.cluster import KMeans as _KMeans
import numpy as np


# In[0]
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans2 = _KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)
print(kmeans.labels_)
print(kmeans.predict([[0, 0], [12, 3]]))
print(kmeans.cluster_centers_)
print()
kmeans2.fit(X)
print(kmeans2.labels_)
print(kmeans2.predict([[0, 0], [12, 3]]))
print(kmeans2.cluster_centers_)
print()
"""Out[0]
tensor([0, 0, 0, 1, 1, 1])
tensor([0, 1])
tensor([[ 1.,  2.],
        [10.,  2.]])

[1 1 1 0 0 0]
[1 0]
[[10.  2.]
 [ 1.  2.]]
"""

# In[1] test gpu
kmeans = KMeans(n_clusters=2, random_state=42, device='cuda')
kmeans.fit(X)
print(kmeans.labels_)
print(kmeans.predict([[0, 0], [12, 3]]))
print(kmeans.cluster_centers_)
"""Out[1]
tensor([1, 1, 1, 0, 0, 0], device='cuda:0')
tensor([1, 0], device='cuda:0')
tensor([[10.,  2.],
        [ 1.,  2.]], device='cuda:0')
"""