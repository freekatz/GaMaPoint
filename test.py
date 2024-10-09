import torch
from pykdtree.kdtree import KDTree

xyz = torch.randn((80000, 3))
xyz_c = xyz.clone()
alpha = 0.2
p_pow = (xyz - xyz.min(0)[0]).pow(2).sum(dim=-1)
scale_max = p_pow.max(0)[0]
scale_min = p_pow.min(0)[0]
scaled_alpha = alpha * (scale_max - scale_min)
xyz = xyz.detach().numpy()
visible = torch.randn(80000, 16)
# visible = (visible != 0).int()
visible = visible.detach().numpy()
import time
start_time = time.time()
kd_tree = KDTree(xyz, visible)
dist, idx = kd_tree.query(xyz, visible, k=8, alpha=0)
dist = torch.from_numpy(dist)
idx = torch.from_numpy(idx).long()
xyz = torch.from_numpy(xyz)
visible = torch.from_numpy(visible)
end_time = time.time()
print(end_time - start_time)
print(dist.shape, idx.shape)
print(dist)
print(idx)

xyz_c = xyz_c.detach().numpy()
visible = visible.detach().numpy()
start_time = time.time()
kd_tree = KDTree(xyz_c, visible)
dist, idx2 = kd_tree.query(xyz_c, visible, k=8, alpha=alpha)
dist = torch.from_numpy(dist)
idx2 = torch.from_numpy(idx2).long()
xyz_c = torch.from_numpy(xyz_c)
visible = torch.from_numpy(visible)
end_time = time.time()
print(end_time - start_time)
print(dist.shape, idx.shape)
print(dist)
print(idx==idx2)
