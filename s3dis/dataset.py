import __init__

import torch
import random
import math
from torch.utils.data import Dataset
from pathlib import Path

from backbone.ops.gs_3d import NaiveGaussian3D, GaussianOptions
from utils.cutils import grid_subsampling, KDTree, grid_subsampling_test


class S3DIS(Dataset):
    classes = ['ceiling',
               'floor',
               'wall',
               'beam',
               'column',
               'window',
               'door',
               'chair',
               'table',
               'bookcase',
               'sofa',
               'board',
               'clutter']
    num_classes = 13
    class2color = {'ceiling': [0, 255, 0],
                   'floor': [0, 0, 255],
                   'wall': [0, 255, 255],
                   'beam': [255, 255, 0],
                   'column': [255, 0, 255],
                   'window': [100, 100, 255],
                   'door': [200, 200, 100],
                   'table': [170, 120, 200],
                   'chair': [255, 0, 0],
                   'sofa': [200, 100, 100],
                   'bookcase': [10, 200, 100],
                   'board': [200, 200, 200],
                   'clutter': [50, 50, 50]}
    cmap = [*class2color.values()]

    def __init__(self,
                 dataset_dir: Path,
                 area="!5",
                 loop=30,
                 train=True,
                 warmup=False,
                 k=[24, 24, 24, 24],
                 grid_size=[0.04, 0.08, 0.16, 0.32],
                 voxel_max=24000,
                 batch_size=8,
                 gs_opts: GaussianOptions = GaussianOptions.default(),
                 ):
        dataset_dir = Path(dataset_dir)
        self.data_paths = list(dataset_dir.glob(f'[{area}]*.pt'))
        self.loop = loop
        self.train = train
        self.warmup = warmup
        self.k = k
        self.grid_size = grid_size
        self.voxel_max = voxel_max
        self.batch_size = batch_size
        self.gs_opts = gs_opts

        assert len(self.data_paths) > 0
        assert len(self.k) == len(self.grid_size)

        if train and warmup:
            max_n = 0
            selected_data = self.data_paths[0]
            for data in self.data_paths:
                n = torch.load(data)[0].shape[0]
                if n > max_n:
                    max_n = n
                    selected_data = data
            # use selected data with max n to warmup model
            self.data_paths = [selected_data]

        self.datas = [torch.load(path) for path in self.data_paths]

    def __len__(self):
        return len(self.data_paths) * self.loop

    def __getitem__(self, idx):
        if not self.train:
            return self.__getitem_test__(idx)

        idx = idx // self.loop
        xyz, colors, label = self.datas[idx]

        # xyz transforms
        # todo remove
        if self.train:
            xyz = self.xyz_transform(xyz)

        # pre sample xyz
        # here grid size is assumed 0.04, so estimated downsampling ratio is ~14
        if self.train:
            ds_idx = grid_subsampling(xyz, self.grid_size[0], 2.5 / 14)
        else:
            ds_idx = grid_subsampling_test(xyz, self.grid_size[0], 2.5 / 14, pick=0)
        xyz = xyz[ds_idx]
        # todo remove
        if not self.train:
            xyz -= xyz.min(dim=0)[0]

        # select max voxel by random
        if xyz.shape[0] > self.voxel_max and self.train:
            pt = random.choice(xyz)
            condition = (xyz - pt).square().sum(dim=1).argsort()[:self.voxel_max].sort()[0]  # sort to preserve locality
            xyz = xyz[condition]
            ds_idx = ds_idx[condition]
        colors = colors[ds_idx].float()
        label = label[ds_idx]
        # random to fill color with zero
        colors = self.color_transform(colors)
        # mix z into feature
        height = xyz[:, 2:]
        feature = torch.cat([colors, height], dim=1)
        # gs projects, sampling, grouping and make points
        gs = self.make_points(xyz, idx)
        gs.gs_points.__update_attr__('f', feature)
        # todo remove
        xyz.mul_(40)
        gs.gs_points.__update_attr__('p', xyz)
        gs.gs_points.__update_attr__('y', label)
        return gs

    def __getitem_test__(self, idx):
        idx_pick = idx % self.loop * 5
        idx = idx // self.loop
        xyz, colors, label = self.datas[idx]
        full_xyz = xyz
        full_label = label

        # pre sample xyz
        idx_sampled = grid_subsampling_test(xyz, self.grid_size[0], 2.5 / 14, pick=idx_pick)
        xyz = xyz[idx_sampled]
        colors = colors[idx_sampled].float()
        full_label = full_label[idx_sampled]
        full_nn = KDTree(xyz).knn(full_xyz, 1)[0].squeeze(-1)

        xyz -= xyz.min(dim=0)[0]
        colors.mul_(1 / 250.)
        feature = torch.cat([colors, xyz[:, 2:]], dim=1)

        # gs projects, sampling, grouping and make points
        gs = self.make_points(xyz, idx)
        gs.gs_points.__update_attr__('f', feature)
        # todo remove
        xyz.mul_(40)
        gs.gs_points.__update_attr__('p', xyz)
        gs.gs_points.__update_attr__('y', full_label)
        gs.gs_points.__update_attr__('pts_list', full_nn)
        return gs

    def xyz_transform(self, xyz):
        angle = random.random() * 2 * math.pi
        cos, sin = math.cos(angle), math.sin(angle)
        rotmat = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
        rotmat *= random.uniform(0.8, 1.2)
        xyz = xyz @ rotmat
        xyz += torch.empty_like(xyz).normal_(std=0.005)
        xyz -= xyz.min(dim=0)[0]
        return xyz

    def color_transform(self, colors):
        if self.train and random.random() < 0.2:
            # todo remove
            colors.fill_(0.)
        else:
            # color transforms
            if self.train and random.random() < 0.2:
                colmin = colors.min(dim=0, keepdim=True)[0]
                colmax = colors.max(dim=0, keepdim=True)[0]
                scale = 255 / (colmax - colmin)
                alpha = random.random()
                colors = (1 - alpha + alpha * scale) * colors - alpha * colmin * scale
            colors.mul_(1 / 250.)
        return colors

    def make_points(self, p, idx) -> NaiveGaussian3D:
        n_layers = len(self.grid_size)
        full_xyz = p
        gs = NaiveGaussian3D(self.gs_opts, batch_size=self.batch_size)
        # gs.projects(p, cam_seed=idx, scale=2*40.)
        gs.projects(p, cam_seed=idx)
        p_gs = gs.gs_points.p_gs

        idx_ds = []
        idx_us = []
        idx_group = []
        idx_gs_group = []
        for i in range(n_layers):
            # down sample
            if i > 0:
                grid_size = self.grid_size[i]
                if self.warmup:
                    ds_idx = torch.randperm(p.shape[0])[:int(p.shape[0] / grid_size)].contiguous()
                else:
                    ds_idx = grid_subsampling(p, grid_size)
                p = p[ds_idx]
                p_gs = p_gs[ds_idx]
                idx_ds.append(ds_idx)

            # group
            k = self.k[i]
            kdt = KDTree(p)
            idx_group.append(kdt.knn(p, k, False)[0].long())
            # kdt_gs = KDTree(p_gs)
            # idx_gs_group.append(kdt_gs.knn(p_gs, k, False)[0].long())

            # up sample
            if i > 0:
                us_idx = kdt.knn(full_xyz, 1, False)[0].squeeze(-1)
                idx_us.append(us_idx)

        gs.gs_points.__update_attr__('idx_ds', idx_ds)
        gs.gs_points.__update_attr__('idx_us', idx_us)
        gs.gs_points.__update_attr__('idx_group', idx_group)
        gs.gs_points.__update_attr__('idx_gs_group', idx_gs_group)
        return gs


def s3dis_collate_fn(batch):
    gs_list = list(batch)
    new_gs = NaiveGaussian3D(gs_list[0].opt, batch_size=gs_list[0].batch_size)
    new_gs = merge_gs_list(new_gs, gs_list)
    return new_gs


def s3dis_collate_test_fn(batch):
    return batch[0]


def merge_gs_list(new_gs, gs_list):
    p_all = []
    p_gs_all = []
    f_all = []
    y_all = []
    idx_ds_all = []
    idx_us_all = []
    idx_group_all = []
    idx_gs_group_all = []
    pts_all = []
    n_layers = len(gs_list[0].gs_points.idx_group)
    pts_per_layer = [0] * n_layers
    for i in range(len(gs_list)):
        gs = gs_list[i]
        p_all.append(gs.gs_points.p)
        p_gs_all.append(gs.gs_points.p_gs)
        f_all.append(gs.gs_points.f)
        y_all.append(gs.gs_points.y)

        idx_ds = gs.gs_points.idx_ds
        idx_us = gs.gs_points.idx_us
        idx_group = gs.gs_points.idx_group
        idx_gs_group = gs.gs_points.idx_gs_group
        pts = []
        for layer_idx in range(n_layers):
            if layer_idx < len(idx_us):
                idx_ds[layer_idx].add_(pts_per_layer[layer_idx])
                idx_us[layer_idx].add_(pts_per_layer[layer_idx+1])
            idx_group[layer_idx].add_(pts_per_layer[layer_idx])
            # idx_gs_group[layer_idx].add_(pts_per_layer[layer_idx])
            pts.append(idx_group[layer_idx].shape[0])
        idx_ds_all.append(idx_ds)
        idx_us_all.append(idx_us)
        idx_group_all.append(idx_group)
        idx_gs_group_all.append(idx_gs_group)
        pts_all.append(pts)
        pts_per_layer = [pt + idx.shape[0] for (pt, idx) in zip(pts_per_layer, idx_group)]

    p = torch.cat(p_all, dim=0)
    p_gs = torch.cat(p_gs_all, dim=0)
    f = torch.cat(f_all, dim=0)
    y = torch.cat(y_all, dim=0)
    idx_ds = [torch.cat(idx, dim=0) for idx in zip(*idx_ds_all)]
    idx_us = [torch.cat(idx, dim=0) for idx in zip(*idx_us_all)]
    idx_group = [torch.cat(idx, dim=0) for idx in zip(*idx_group_all)]
    idx_gs_group = [torch.cat(idx, dim=0) for idx in zip(*idx_gs_group_all)]
    new_gs.gs_points.__update_attr__('p', p)
    new_gs.gs_points.__update_attr__('p_gs', p_gs)
    new_gs.gs_points.__update_attr__('f', f)
    new_gs.gs_points.__update_attr__('y', y)
    new_gs.gs_points.__update_attr__('idx_ds', idx_ds)
    new_gs.gs_points.__update_attr__('idx_us', idx_us)
    new_gs.gs_points.__update_attr__('idx_group', idx_group)
    new_gs.gs_points.__update_attr__('idx_gs_group', idx_gs_group)
    pts_list = torch.tensor(pts_all, dtype=torch.int64).view(-1, n_layers).transpose(0, 1).contiguous()
    new_gs.gs_points.__update_attr__('pts_list', pts_list)
    return new_gs
