import __init__

import torch

from pykdtree.kdtree import KDTree

from backbone.gs_3d import NaiveGaussian3D, GaussianOptions
from utils import points_scaler, vis_multi_points
from utils.binary import bin2dec_split
from utils.subsample import fps_sample
from utils.vis_3d import vis_knn, vis_knn2, vis_knn3, vis_knn4, vis_labels


def analyse():
    presample_path = './dataset_link/shapenetpart_presample.pt'
    xyz, norm, shape, seg = torch.load(presample_path)

    # shape_id = 10
    alpha = 0.1
    gs = NaiveGaussian3D(GaussianOptions.default(), batch_size=1, device=xyz.device)
    gs.opt.n_cameras = 64
    gs.opt.cam_fovy = 120
    k = 64
    n_samples = 256
    # p_idx = 79

    # bike_id = [3, 17, 26, 27, 28, 33, 35, 37, 40, 43, 49, 50]
    # bike_id = [3, 49, 50]
    bike_id = [3]
    # 10-3-191

    # idx_all = torch.nonzero((shape == shape_id).int())
    # for i in bike_id:
    #     idx = idx_all[i]  # 23-156 19-15 46-15
    #     p = xyz[idx]
    #     y = seg[idx].squeeze(0)
    #     s = shape[idx]
    #     p, ds_idx = fps_sample(p, n_samples)
    #     N = p.shape[0]
    #     p = points_scaler(p, scale=1.0)
    #     p = p.squeeze(0)
    #     gs.projects(p, cam_seed=1, cam_batch=gs.opt.n_cameras * 2)
    #     y = y[ds_idx.squeeze(0)]
    #
    #     ps, _ = fps_sample(p.unsqueeze(0), 2, random_start_point=True)
    #     ps = ps.squeeze(0)
    #     p0, p1 = ps[0], ps[1]
    #     scaler = (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 + (p0[2] - p1[2]) ** 2
    #     visible = gs.gs_points.visible.squeeze(1).float()
    #     # v = bin2dec_split(visible, max_bits=32)  # N x M//64
    #
    #     kdt_1 = KDTree(p.detach().cpu().numpy(), visible.detach().cpu().numpy())
    #     _, group_idx_1 = kdt_1.query(p.detach().cpu().numpy(), visible.detach().cpu().numpy(), k=k, alpha=0.)
    #     group_idx_1 = torch.from_numpy(group_idx_1)
    #
    #     # vis_knn(p, 10, group_idx_1, title=f'{shape_id}-{i}', point_size=18)
    #
    #     # _, group_idx_2 = kdt_1.query(p.detach().cpu().numpy(), visible.detach().cpu().numpy(), k=k, alpha=alpha,
    #     #                              scaler=scaler)
    #     # group_idx_2 = torch.from_numpy(group_idx_2)
    #
    #     _, group_idx_3 = kdt_1.query(p.detach().cpu().numpy(), visible.detach().cpu().numpy(), k=k, alpha=-1)
    #     group_idx_3 = torch.from_numpy(group_idx_3)
    #
    #     # vis_knn2(p, 156, group_idx_1, group_idx_2, point_size=18)
    #     # vis_knn3(p, 10, group_idx_1, group_idx_2, point_size=18)
    #     for p_idx in range(512):
    #         vis_knn2(p, p_idx, group_idx_1, group_idx_3, point_size=18, title=f'{p_idx}')
    #
    #     # vis_knn4(p, y, p_idx, group_idx_1, group_idx_3,  point_size=18)


    r1n,w1n= 0, 0
    r2n,w2n= 0, 0
    max_diff_n = 0
    max_p_idx_n = 0
    # for shape_id in range(16):
    for shape_id in [0]:
        idx_all = torch.nonzero((shape == shape_id).int())
        for i in range(idx_all.shape[0]):
        # for i in [14]:
            idx = idx_all[i]  # 23-156 19-15 46-15
            p = xyz[idx]
            y = seg[idx].squeeze(0)
            s = shape[idx]
            p, ds_idx = fps_sample(p, n_samples)
            N = p.shape[0]
            p = points_scaler(p, scale=1.0)
            p = p.squeeze(0)
            gs.projects(p, cam_seed=1, cam_batch=gs.opt.n_cameras * 2)
            y = y[ds_idx.squeeze(0)]

            ps, _ = fps_sample(p.unsqueeze(0), 2, random_start_point=True)
            ps = ps.squeeze(0)
            p0, p1 = ps[0], ps[1]
            scaler = (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 + (p0[2] - p1[2]) ** 2
            visible = gs.gs_points.visible.squeeze(1).float()

            kdt_1 = KDTree(p.detach().cpu().numpy(), visible.detach().cpu().numpy())
            _, group_idx_1 = kdt_1.query(p.detach().cpu().numpy(), visible.detach().cpu().numpy(), k=k, alpha=0.)
            group_idx_1 = torch.from_numpy(group_idx_1)

            # vis_knn(p, 10, group_idx_1, title=f'{shape_id}-{i}', point_size=18)

            # _, group_idx_2 = kdt_1.query(p.detach().cpu().numpy(), visible.detach().cpu().numpy(), k=k, alpha=alpha,
            #                              scaler=scaler)
            # group_idx_2 = torch.from_numpy(group_idx_2)

            _, group_idx_2 = kdt_1.query(p.detach().cpu().numpy(), visible.detach().cpu().numpy(), k=k, alpha=-1)
            group_idx_2 = torch.from_numpy(group_idx_2)

            lg1 = (y[group_idx_1] - y.unsqueeze(-1)) != 0
            lg2 = (y[group_idx_2] - y.unsqueeze(-1)) != 0
            lg1 = lg1.sum(-1)
            lg2 = lg2.sum(-1)
            res_idx = torch.nonzero(lg1 > lg2).squeeze()
            res_idx2 = torch.nonzero(lg1 == lg2).squeeze()
            res_idx3 = torch.nonzero(lg1 < lg2).squeeze()
            # print(res_idx.shape, res_idx2.shape, res_idx3.shape)

            max_diff = 0
            max_p_idx = 0
            if len(res_idx.shape) == 0 or res_idx.shape[0] == 0:
                continue
            for p_idx in res_idx:
                _, _, r1, w1, r2, w2 = vis_knn4(p, y, p_idx, group_idx_1, group_idx_2, vis=False,  point_size=24)
                if w1 > w2:
                    if w1 - w2 > max_diff:
                        max_diff = w1 - w2
                        max_p_idx = p_idx
                    if w1 - w2 > 30 :
                        print(torch.unique(y))
                        print(f'shape: {shape_id}, i: {i}, p_idx: {p_idx}, xyz: {p[p_idx]}, r1: {r1}, r2: {r2}, w1: {w1}, w2: {w2}, diff: {w1 - w2}')
                        vis_knn2(p, p_idx, group_idx_1, group_idx_2, point_size=24, title=f'{shape_id}-{i}-{p_idx}')
                        vis_knn4(p, y, p_idx, group_idx_1, group_idx_2, point_size=24, title=f'{shape_id}-{i}-{p_idx}')
            # print(max_diff, max_p_idx)
            if max_diff > max_diff_n:
                print(f'shape: {shape_id}, i: {i}, p_idx: {max_p_idx}, xyz: {p[max_p_idx]}')
                max_diff_n = max_diff
                max_p_idx_n = max_p_idx
                print(max_diff_n, max_p_idx_n)
    print(max_diff_n, max_p_idx_n)
    # print(w1n/(r1n+w1n), w2n/(r2n+w2n))


def visual():
    # 4 167 224
    # 7 14 0
    # 7 45 116
    # 8 51 125
    # 8 143 91
    # 0 82 216
    # 0 315 12
    shape_id = [0, 4, 7, 8]
    obj_id = [315, 167, 14, 51]
    p_idx = [12, 224, 0, 125]

    presample_path = './dataset_link/shapenetpart_presample.pt'
    xyz, norm, shape, seg = torch.load(presample_path)

    # alpha = 0.1
    gs = NaiveGaussian3D(GaussianOptions.default(), batch_size=1, device=xyz.device)
    gs.opt.n_cameras = 64
    gs.opt.cam_fovy = 120
    k = 64
    n_samples = 256

    ps = []
    cs = []
    for i in range(len(shape_id)):
        s_id = shape_id[i]
        o_id = obj_id[i]
        p_id = p_idx[i]
        idx_all = torch.nonzero((shape == s_id).int())
        idx = idx_all[o_id]  # 23-156 19-15 46-15
        p = xyz[idx]
        y = seg[idx].squeeze(0)
        p, ds_idx = fps_sample(p, n_samples)
        p = points_scaler(p, scale=1.0)
        p = p.squeeze(0)
        gs.projects(p, cam_seed=1, cam_batch=gs.opt.n_cameras * 2)
        y = y[ds_idx.squeeze(0)]
        visible = gs.gs_points.visible.squeeze(1).float()

        kdt_1 = KDTree(p.detach().cpu().numpy(), visible.detach().cpu().numpy())
        _, group_idx_1 = kdt_1.query(p.detach().cpu().numpy(), visible.detach().cpu().numpy(), k=k, alpha=0.)
        group_idx_1 = torch.from_numpy(group_idx_1)
        _, group_idx_2 = kdt_1.query(p.detach().cpu().numpy(), visible.detach().cpu().numpy(), k=k, alpha=-1)
        group_idx_2 = torch.from_numpy(group_idx_2)

        c0 = vis_labels(p, y, vis=False)
        c1, c2, r1, w1, r2, w2 = vis_knn4(p, y, p_id, group_idx_1, group_idx_2, vis=False,  point_size=24)
        ps.append(p.detach().cpu().numpy())
        ps.append(p.detach().cpu().numpy())
        ps.append(p.detach().cpu().numpy())
        cs.append(c0.detach().cpu().numpy())
        cs.append(c1.detach().cpu().numpy())
        cs.append(c2.detach().cpu().numpy())
    vis_multi_points(ps, cs, plot_shape=(len(shape_id), 3), point_size=24)


if __name__ == '__main__':
    # analyse()
    visual()


