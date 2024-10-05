import math
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from backbone.gs_3d import GaussianOptions, NaiveGaussian3D, make_gs_points, merge_gs_list
from utils.cutils import grid_subsampling, grid_subsampling_test


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
                 voxel_max=24000,
                 k=[16, 16, 16, 16],
                 k_gs=[4, 4, 4, 4],
                 grid_size=[0.08, 0.16, 0.32],
                 visible_sample_stride=0.,
                 batch_size=8,
                 gs_opts: GaussianOptions = GaussianOptions.default(),
                 ):
        dataset_dir = Path(dataset_dir)
        self.data_paths = list(dataset_dir.glob(f'[{area}]*.pt'))
        self.loop = loop
        self.train = train
        self.warmup = warmup
        self.voxel_max = voxel_max
        self.k = k
        self.k_gs = k_gs
        self.grid_size = grid_size
        self.visible_sample_stride = visible_sample_stride
        self.batch_size = batch_size
        self.gs_opts = gs_opts

        assert len(self.data_paths) > 0

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

    @classmethod
    def get_classes(cls):
        return cls.classes

    def __getitem__(self, idx):
        if not self.train:
            return self.get_test_item(idx)

        idx //= self.loop
        xyz, col, lbl = self.datas[idx]

        if self.train:
            angle = random.random() * 2 * math.pi
            cos, sin = math.cos(angle), math.sin(angle)
            rotmat = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
            rotmat *= random.uniform(0.8, 1.2)
            xyz = xyz @ rotmat
            xyz += torch.empty_like(xyz).normal_(std=0.005)
            xyz -= xyz.min(dim=0)[0]

        # here grid size is assumed 0.04, so estimated downsampling ratio is ~14
        if self.train:
            indices = grid_subsampling(xyz, 0.04, 2.5 / 14)
        else:
            indices = grid_subsampling_test(xyz, 0.04, 2.5 / 14, pick=0)

        xyz = xyz[indices]

        if not self.train:
            xyz -= xyz.min(dim=0)[0]

        if xyz.shape[0] > self.voxel_max and self.train:
            pt = random.choice(xyz)
            condition = (xyz - pt).square().sum(dim=1).argsort()[:self.voxel_max].sort()[0]  # sort to preserve locality
            xyz = xyz[condition]
            indices = indices[condition]

        col = col[indices]
        lbl = lbl[indices]
        col = col.float()

        if self.train and random.random() < 0.2:
            col.fill_(0.)
        else:
            if self.train and random.random() < 0.2:
                colmin = col.min(dim=0, keepdim=True)[0]
                colmax = col.max(dim=0, keepdim=True)[0]
                scale = 255 / (colmax - colmin)
                alpha = random.random()
                col = (1 - alpha + alpha * scale) * col - alpha * colmin * scale
            col.mul_(1 / 250.)

        height = xyz[:, 2:]
        feature = torch.cat([col, height], dim=1)

        gs = NaiveGaussian3D(self.gs_opts, batch_size=self.batch_size, device=xyz.device)
        gs.gs_points.__update_attr__('p', xyz)
        gs.gs_points.__update_attr__('f', feature)
        gs.gs_points.__update_attr__('y', lbl)
        gs.projects(xyz, cam_seed=idx, cam_batch=gs.opt.n_cameras*2)
        gs.gs_points = make_gs_points(gs.gs_points, self.k, self.k_gs, self.grid_size, None,
                                      up_sample=True, visible_sample_stride=self.visible_sample_stride)
        return gs

    def get_test_item(self, idx):
        pick = idx % self.loop * 5

        idx //= self.loop
        xyz, col, lbl = self.datas[idx]

        indices = grid_subsampling_test(xyz, 0.04, 2.5 / 14, pick=pick)
        xyz = xyz[indices]
        lbl = lbl[indices]
        col = col[indices].float()
        col.mul_(1 / 250.)

        xyz -= xyz.min(dim=0)[0]
        feature = torch.cat([col, xyz[:, 2:]], dim=1)

        gs = NaiveGaussian3D(self.gs_opts, batch_size=self.batch_size, device=xyz.device)
        gs.gs_points.__update_attr__('p', xyz)
        gs.gs_points.__update_attr__('f', feature)
        gs.gs_points.__update_attr__('y', lbl)
        gs.projects(xyz, cam_seed=idx, cam_batch=gs.opt.n_cameras*2)
        gs.gs_points = make_gs_points(gs.gs_points, self.k, self.k_gs, self.grid_size, None,
                                      up_sample=True, visible_sample_stride=self.visible_sample_stride)
        return gs



def s3dis_collate_fn(batch):
    gs_list = list(batch)
    new_gs = merge_gs_list(gs_list)
    return new_gs
