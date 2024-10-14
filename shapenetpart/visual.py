import math

import __init__

import torch

from pykdtree.kdtree import KDTree

from backbone.gs_3d import NaiveGaussian3D, GaussianOptions
from utils import points_scaler
from utils.binary import bin2dec_split
from utils.subsample import fps_sample
from utils.vis_3d import vis_knn, vis_knn2, vis_knn3

if __name__ == '__main__':
    presample_path = './dataset_link/shapenetpart_presample.pt'
    xyz, norm, shape, seg = torch.load(presample_path)
    idx_all = torch.nonzero((shape == 10).int())
    idx = idx_all[23]  # 23-156 19-15 46-15
    p = xyz[idx]
    s = shape[idx]
    print(s)
    p, _ = fps_sample(p, 512)
    N = p.shape[0]
    p = points_scaler(p, scale=1.0)
    p = p.squeeze(0)

    alpha = 0.5
    gs = NaiveGaussian3D(GaussianOptions.default(), batch_size=1, device=p.device)
    gs.opt.n_cameras = 32
    gs.opt.cam_fovy = 120
    gs.projects(p, cam_seed=1, cam_batch=gs.opt.n_cameras * 2)

    ps, _ = fps_sample(p.unsqueeze(0), 2, random_start_point=True)
    ps = ps.squeeze(0)
    p0, p1 = ps[0], ps[1]
    scaler = (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 + (p0[2] - p1[2]) ** 2
    depths = gs.gs_points.depths.squeeze(1).float()
    # v = bin2dec_split(depths, max_bits=32)  # N x M//64

    kdt_1 = KDTree(p.detach().cpu().numpy(), depths.detach().cpu().numpy())
    _, group_idx_1 = kdt_1.query(p.detach().cpu().numpy(), depths.detach().cpu().numpy(), k=132, alpha=0.)
    group_idx_1 = torch.from_numpy(group_idx_1)

    _, group_idx_2 = kdt_1.query(p.detach().cpu().numpy(), depths.detach().cpu().numpy(), k=132, alpha=alpha, scaler=scaler)
    group_idx_2 = torch.from_numpy(group_idx_2)

    _, group_idx_3 = kdt_1.query(p.detach().cpu().numpy(), depths.detach().cpu().numpy(), k=132, alpha=-1)
    group_idx_3 = torch.from_numpy(group_idx_3)

    # vis_knn(p, 10, group_idx_1)
    vis_knn2(p, 156, group_idx_1, group_idx_2, point_size=24)
    # vis_knn3(p, 10, group_idx_1, group_idx_2)

    vis_knn2(p, 156, group_idx_1, group_idx_3, point_size=24)
