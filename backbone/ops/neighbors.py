from __future__ import division

import numpy as np
import torch
from sklearn.metrics.pairwise import paired_distances
from sklearn.neighbors import BallTree


def eucli_dist(x, y):
    return np.sqrt(sum(np.power((x - y), 2)))


def manha_dist(x, y):
    return np.sum(np.abs(x - y))


def cosine_dist(x, y):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    return paired_distances(x, y, metric='cosine')[0]


def bin_dist(x, y):
    x = (x != 0).astype(np.int8)
    y = (y != 0).astype(np.int8)
    return np.mean(x ^ y)


def build_dist_fn(d1_fn=eucli_dist, d2_fn=bin_dist, alpha=0.):

    def fn(x, y):
        x1, y1 = x[:3], y[:3]
        d1 = d1_fn(x1, y1)
        if alpha == 0. or x.shape[0] <= 3:
            return d1
        x2, y2 = x[3:], y[3:]
        d2 = d2_fn(x2, y2)
        return d1 + alpha * d2

    return fn


def build_tree(x, metric=build_dist_fn()):
    return BallTree(x, metric=metric)


def knn(tree, query, k, return_distance=False):
    idx = tree.query(query, k=k, return_distance=return_distance)
    return torch.from_numpy(idx)


if __name__ == '__main__':
    X = torch.tensor([[-1, -1, -1], [-2, -1, -2], [-3, -2, -3], [1, 1, 1], [2, 1, 2], [3, 2, 3]])
    depths = torch.randn(6, 16)
    code = (depths != 0).int()
    all_x = torch.cat([X, code], dim=-1)

    import time
    start_time = time.time()
    tree = BallTree(all_x, metric=build_dist_fn(alpha=0.))
    res = knn(tree, all_x, k=2, return_distance=False)
    end_time = time.time()
    print(end_time - start_time)
    print(res.shape)
    Y = X[res]
    print(Y)
    x = np.array([0, 1, 1, 1])
    y = np.array([1, 1, 0, 1])
    d = bin_dist(x, y)
    print(d)
    d = manha_dist(x, y)
    print(d)
    d = cosine_dist(x, y)
    print(d)
    d = eucli_dist(x, y)
    print(d)