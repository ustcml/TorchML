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
print(kmeans.transform(X))
print(kmeans.predict([[0, 0], [12, 3]]))
print(kmeans.labels_)
print(kmeans.cluster_centers_)
print(kmeans.inertia_)

print()
kmeans2.fit(X)
print(kmeans.transform(X))
print(kmeans2.predict([[0, 0], [12, 3]]))
print(kmeans2.labels_)
print(kmeans2.cluster_centers_)
print(kmeans2.inertia_)
print()
"""Out[0]
tensor([[0.0000, 9.0000],
        [2.0000, 9.2195],
        [2.0000, 9.2195],
        [9.0000, 0.0000],
        [9.2195, 2.0000],
        [9.2195, 2.0000]])
tensor([0, 1])
tensor([0, 0, 0, 1, 1, 1])
tensor([[ 1.,  2.],
        [10.,  2.]])
16.0

tensor([[0.0000, 9.0000],
        [2.0000, 9.2195],
        [2.0000, 9.2195],
        [9.0000, 0.0000],
        [9.2195, 2.0000],
        [9.2195, 2.0000]])
[1 0]
[1 1 1 0 0 0]
[[10.  2.]
 [ 1.  2.]]
16.0
"""

# In[1] test gpu
kmeans = KMeans(n_clusters=2, random_state=42, device='cuda')
print(kmeans.fit_predict(X))
print(kmeans.fit_transform(X))
print(kmeans.predict([[0, 0], [12, 3]]))
print(kmeans.labels_)
print(kmeans.cluster_centers_)
print(kmeans.inertia_)
"""Out[1]
tensor([1, 1, 1, 0, 0, 0], device='cuda:0')
tensor([[9.0000, 0.0000],
        [9.2195, 2.0000],
        [9.2195, 2.0000],
        [0.0000, 9.0000],
        [2.0000, 9.2195],
        [2.0000, 9.2195]], device='cuda:0')
tensor([1, 0], device='cuda:0')
tensor([1, 1, 1, 0, 0, 0], device='cuda:0')
tensor([[10.,  2.],
        [ 1.,  2.]], device='cuda:0')
16.0
"""
