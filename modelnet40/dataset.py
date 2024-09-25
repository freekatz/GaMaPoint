import h5py

import __init__

import torch
import random
import math
from torch.utils.data import Dataset
from pathlib import Path

from backbone.gs_3d import GaussianOptions, NaiveGaussian3D, make_gs_points, merge_gs_list, fps_sample
from utils.cutils import grid_subsampling, grid_subsampling_test


class ModelNet40(Dataset):
    def __init__(self,
                 dataset_dir: Path,
                 train=True,
                 warmup=False,
                 voxel_max=1024,
                 k=[20, 20, 20],
                 strides=[1, 4, 4],
                 batch_size=32,
                 gs_opts: GaussianOptions = GaussianOptions.default(),
                 ):
        dataset_dir = Path(dataset_dir)

        self.data_paths = dataset_dir.glob(f"ply_data_{'train' if train else 'test'}*.h5")
        self.train = train
        self.warmup = warmup
        self.voxel_max = voxel_max
        self.k = k
        self.strides = strides
        self.batch_size = batch_size
        self.gs_opts = gs_opts

        datas, label = [], []
        for p in self.data_paths:
            f = h5py.File(p, 'r')
            datas.append(torch.from_numpy(f['data'][:]).float())
            label.append(torch.from_numpy(f['label'][:]).long())
            f.close()
        self.datas = torch.cat(datas)
        self.label = torch.cat(label).squeeze()

    def __len__(self):
        return self.datas.shape[0]

    def __getitem__(self, idx):
        xyz, ds_idx = fps_sample(self.datas[idx], self.voxel_max)
        label = self.label[idx][ds_idx]

        gs = NaiveGaussian3D(self.gs_opts, batch_size=self.batch_size, device=xyz.device)
        gs.gs_points.__update_attr__('p', xyz)
        gs.gs_points.__update_attr__('y', label)
        gs.projects(xyz, cam_seed=idx, cam_batch=gs.opt.n_cameras)
        gs.gs_points = make_gs_points(gs.gs_points, self.k, None, self.strides, up_sample=False)
        return gs

def modelnet40_collate_fn(batch):
    gs_list = list(batch)
    new_gs = merge_gs_list(gs_list)
    return new_gs
