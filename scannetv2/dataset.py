import numpy as np
import scipy

import __init__

import torch
import random
import math
from torch.utils.data import Dataset
from pathlib import Path

from backbone.gs_3d import GaussianOptions, NaiveGaussian3D, make_gs_points, merge_gs_list
from utils.cutils import grid_subsampling, grid_subsampling_test


train_file = Path(__file__).parent / "scannetv2_train.txt"
val_file = Path(__file__).parent / "scannetv2_val.txt"
with open(train_file, 'r') as file:
    scan_train_list = [line.strip() for line in file.readlines()]
with open(val_file, 'r') as file:
    scan_val_list = [line.strip() for line in file.readlines()]


# adapted from https://github.com/Gofinge/PointTransformerV2/blob/main/pcr/datasets/transform.py
class ElasticDistortion(object):
    def __init__(self, distortion_params=None):
        self.distortion_params = [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """
        Apply elastic distortion on sparse coordinate space.
        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
        blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
        blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity *
                                       (noise_dim - 2), noise_dim)
        ]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=False, fill_value=0)
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, coord):
        coord = coord.numpy()
        if random.random() < 0.95:
            for granularity, magnitude in self.distortion_params:
                coord = self.elastic_distortion(coord, granularity, magnitude)
        return torch.from_numpy(coord)


class ScanNetV2(Dataset):
    classes = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
               'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub',
               'otherfurniture']
    class_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39
    ]
    num_classes = 20
    gravity_dim = 2
    color_mean = [0.46259782, 0.46253258, 0.46253258]
    color_std = [0.693565, 0.6852543, 0.68061745]
    cmap = {
        0: (0., 0., 0.),
        1: (174., 199., 232.),
        2: (152., 223., 138.),
        3: (31., 119., 180.),
        4: (255., 187., 120.),
        5: (188., 189., 34.),
        6: (140., 86., 75.),
        7: (255., 152., 150.),
        8: (214., 39., 40.),
        9: (197., 176., 213.),
        10: (148., 103., 189.),
        11: (196., 156., 148.),
        12: (23., 190., 207.),
        14: (247., 182., 210.),
        15: (66., 188., 102.),
        16: (219., 219., 141.),
        17: (140., 57., 197.),
        18: (202., 185., 52.),
        19: (51., 176., 203.),
        20: (200., 54., 131.),
        21: (92., 193., 61.),
        22: (78., 71., 183.),
        23: (172., 114., 82.),
        24: (255., 127., 14.),
        25: (91., 163., 138.),
        26: (153., 98., 156.),
        27: (140., 153., 101.),
        28: (158., 218., 229.),
        29: (100., 125., 154.),
        30: (178., 127., 135.),
        32: (146., 111., 194.),
        33: (44., 160., 44.),
        34: (112., 128., 144.),
        35: (96., 207., 209.),
        36: (227., 119., 194.),
        37: (213., 92., 176.),
        38: (94., 106., 211.),
        39: (82., 84., 163.),
        40: (100., 85., 144.),
    }

    def __init__(self,
                 dataset_dir: Path,
                 loop=6,
                 train=True,
                 warmup=False,
                 voxel_max=64000,
                 k=[24, 24, 24, 24, 24],
                 k_gs=[6, 6, 6, 6, 6],
                 grid_size=[0.04, 0.08, 0.16, 0.32],
                 visible_sample_stride=0.,
                 batch_size=8,
                 gs_opts: GaussianOptions = GaussianOptions.default(),
                 ):
        dataset_dir = Path(dataset_dir)
        data_list = scan_train_list if train else scan_val_list
        self.data_paths = [f"{dataset_dir}/{p}.pt" for p in data_list]
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
        self.els = ElasticDistortion()

    def __len__(self):
        return len(self.data_paths) * self.loop

    def __getitem__(self, idx):
        if not self.train:
            return self.__getitem_test__(idx)

        idx = idx // self.loop
        xyz, colors, norm, label = self.datas[idx]

        # xyz transforms
        if self.train:
            xyz, norm = self.xyz_transform(xyz, norm)

        # pre sample xyz
        if self.train:
            ds_idx = grid_subsampling(xyz, 0.02, 2.5 / 1.5)
        else:
            ds_idx = grid_subsampling_test(xyz, 0.02, 2.5 / 1.5, pick=0)
        xyz = xyz[ds_idx]
        if not self.train:
            xyz -= xyz.min(dim=0)[0]

        # select max voxel by random
        if xyz.shape[0] > self.voxel_max and self.train:
            pt = random.choice(xyz)
            condition = (xyz - pt).square().sum(dim=1).argsort()[:self.voxel_max].sort()[0]  # sort to preserve locality
            xyz = xyz[condition]
            ds_idx = ds_idx[condition]
        colors = colors[ds_idx].float()
        norm = norm[ds_idx]
        label = label[ds_idx]
        # random to fill norm with zero
        norm = self.norm_transform(norm)
        # random to fill color with zero
        colors = self.color_transform(colors)
        # mix z into feature
        height = xyz[:, 2:]
        feature = torch.cat([colors, height, norm], dim=1)

        gs = NaiveGaussian3D(self.gs_opts, batch_size=self.batch_size, device=xyz.device)
        gs.gs_points.__update_attr__('p', xyz)
        gs.gs_points.__update_attr__('f', feature)
        gs.gs_points.__update_attr__('y', label)
        gs.projects(xyz, cam_seed=idx, cam_batch=gs.opt.n_cameras*2)
        gs.gs_points = make_gs_points(gs.gs_points, self.k, self.k_gs, self.grid_size, None, up_sample=True, visible_sample_stride=self.visible_sample_stride)
        return gs

    def __getitem_test__(self, idx):
        rotations = [0, 0.5, 1, 1.5]
        scales = [0.95, 1, 1.05]
        augs = len(rotations) * len(scales)
        aug = idx % self.loop
        idx_pick = aug // augs
        aug %= augs

        idx = idx // self.loop
        xyz, colors, norm, label = self.datas[idx]
        xyz, norm = self.xyz_transform_test(xyz, norm, aug, rotations, scales)

        # pre sample xyz
        idx_sampled = grid_subsampling_test(xyz, 0.02, 2.5 / 1.5, pick=idx_pick)
        xyz = xyz[idx_sampled]
        colors = colors[idx_sampled].float()
        norm = norm[idx_sampled]
        label = label[idx_sampled]

        xyz -= xyz.min(dim=0)[0]
        colors.mul_(1 / 250.)
        feature = torch.cat([colors, xyz[:, 2:], norm], dim=1)

        gs = NaiveGaussian3D(self.gs_opts, batch_size=self.batch_size, device=xyz.device)
        gs.gs_points.__update_attr__('p', xyz)
        gs.gs_points.__update_attr__('f', feature)
        gs.gs_points.__update_attr__('y', label)
        gs.projects(xyz, cam_seed=idx, cam_batch=gs.opt.n_cameras*2)
        gs.gs_points = make_gs_points(gs.gs_points, self.k, self.k_gs, self.grid_size, None, up_sample=True, visible_sample_stride=self.visible_sample_stride)
        return gs

    def xyz_transform(self, xyz, norm):
        angle = random.random() * 2 * math.pi
        cos, sin = math.cos(angle), math.sin(angle)
        rotmat = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
        norm = norm @ rotmat
        rotmat *= random.uniform(0.8, 1.2)
        xyz = xyz @ rotmat
        xyz = self.els(xyz)
        xyz -= xyz.min(dim=0)[0]
        return xyz, norm

    def xyz_transform_test(self, xyz, norm, aug, rotations, scales):
        angle = math.pi * rotations[aug // len(scales)]
        cos, sin = math.cos(angle), math.sin(angle)
        rotmat = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
        norm = norm @ rotmat
        rotmat *= scales[aug % len(scales)]
        xyz = xyz @ rotmat
        xyz -= xyz.min(dim=0)[0]
        return xyz, norm

    def color_transform(self, colors):
        if self.train and random.random() < 0.2:
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

    def norm_transform(self, norm):
        if self.train and random.random() < 0.2:
            norm.fill_(0.)
        return norm


def scannetv2_collate_fn(batch):
    gs_list = list(batch)
    new_gs = merge_gs_list(gs_list)
    return new_gs
