import __init__

import h5py
import torch
from torch.utils.data import Dataset
from pathlib import Path

from backbone.gs_3d import GaussianOptions, NaiveGaussian3D, make_gs_points, merge_gs_list, fps_sample, make_gs_features


class ModelNet40(Dataset):
    def __init__(self,
                 dataset_dir: Path,
                 train=True,
                 warmup=False,
                 num_points=1024,
                 k=[24, 24, 24],
                 k_gs=[6, 6, 6],
                 strides=[1, 4, 4],
                 visible_sample_stride=0.,
                 batch_size=32,
                 gs_opts: GaussianOptions = GaussianOptions.default(),
                 ):
        dataset_dir = Path(dataset_dir)

        self.data_paths = dataset_dir.glob(f"ply_data_{'train' if train else 'test'}*.h5")
        self.train = train
        self.warmup = warmup
        self.num_points = num_points
        self.k = k
        self.k_gs = k_gs
        self.strides = strides
        self.visible_sample_stride = visible_sample_stride
        self.batch_size = batch_size
        self.gs_opts = gs_opts

        datas, label = [], []
        for p in self.data_paths:
            f = h5py.File(p, 'r')
            datas.append(torch.from_numpy(f['data'][:]).float())
            label.append(torch.from_numpy(f['label'][:]).long())
            f.close()
        self.datas = torch.cat(datas)
        self.label = torch.cat(label)

    def __len__(self):
        return self.datas.shape[0]

    def __getitem__(self, idx):
        xyz = self.datas[idx]
        label = self.label[idx]
        if self.train:
            scale = torch.rand((3,)) * (3/2 - 2/3) + 2/3
            xyz = xyz * scale
            xyz = xyz[torch.randperm(xyz.shape[0])]

        xyz, _ = fps_sample(xyz.unsqueeze(0), self.num_points)
        xyz = xyz.squeeze(0)
        height = xyz[:, 2:] * 4
        height -= height.min(dim=0, keepdim=True)[0]
        if self.train:
            height += torch.empty((1, 1)).uniform_(-0.2, 0.2) * 4
        feature = height

        gs = NaiveGaussian3D(self.gs_opts, batch_size=self.batch_size, device=xyz.device)
        gs.gs_points.__update_attr__('p', xyz)
        gs.gs_points.__update_attr__('y', label)
        gs.projects(xyz, cam_seed=idx, cam_batch=gs.opt.n_cameras*2)
        gs.gs_points = make_gs_points(gs.gs_points, self.k, self.k_gs, None, self.strides,
                                      up_sample=False, visible_sample_stride=self.visible_sample_stride)
        # colors = make_gs_features(gs)
        # feature = torch.cat([feature, colors.unsqueeze(-1)], dim=-1)
        gs.gs_points.__update_attr__('f', feature)
        return gs


def modelnet40_collate_fn(batch):
    gs_list = list(batch)
    new_gs = merge_gs_list(gs_list, up_sample=False)
    return new_gs
