from sklearn.neighbors import KDTree
import numpy as np
X = np.array([[-1, -1, -1, -1], [-2, -1, -2, -1], [-3, -2, -3, -2], [1, 1, 1, 1], [2, 1, 2, 1], [3, 2, 3, 2]])
kdt = KDTree(X, leaf_size=30, metric='euclidean')
res = kdt.query(X, k=3, return_distance=False)
Y = X[res]
print(res, Y)
