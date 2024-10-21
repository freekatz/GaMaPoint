import logging

import __init__

import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from backbone.gs_3d import GaussianOptions, NaiveGaussian3D, make_gs_points, merge_gs_list
from utils.cutils import grid_subsampling, grid_subsampling_test


def load_scan_kitti(pc_path):
    scan = np.fromfile(pc_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan


def load_label_kitti(label_path, remap_lut):
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape((-1))
    sem_label = label & 0xFFFF  # semantic label in lower half
    # inst_label = label >> 16  # instance id in upper half
    sem_label = remap_lut[sem_label] - 1
    return sem_label.astype(np.int32)


def get_semantickitti_file_list(dataset_path, test_seq_num):
    seq_list = np.sort(os.listdir(dataset_path))

    train_file_list = []
    test_file_list = []
    val_file_list = []
    for seq_id in seq_list:
        seq_path = os.path.join(dataset_path, seq_id)
        label_path = os.path.join(seq_path, 'labels')
        pc_path = os.path.join(seq_path, 'velodyne')
        path_list = [[os.path.join(pc_path, f), os.path.join(label_path, f.replace('bin', 'label'))] for f in
                     np.sort(os.listdir(pc_path))]

        if seq_id == '08':
            val_file_list.append(path_list)
            if seq_id == test_seq_num:
                test_file_list.append(path_list)
        elif int(seq_id) >= 11 and seq_id == test_seq_num:
            print("\n\n\n Loading test seq_id ", test_seq_num)
            test_file_list.append(path_list)
        elif seq_id in ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']:
            train_file_list.append(path_list)

    train_file_list = np.concatenate(train_file_list, axis=0)
    val_file_list = np.concatenate(val_file_list, axis=0)

    if test_seq_num != 'None':
        test_file_list = np.concatenate(test_file_list, axis=0)
    else:
        test_file_list = None
    return train_file_list, val_file_list, test_file_list


remap_lut_write = np.array([
    0, 10, 11, 15, 18, 20, 30, 31, 32, 40, 44, 48, 49, 50, 51, 70, 71,
    72, 80, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
], dtype=np.int32)
remap_lut_read = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 5, 0, 3, 5, 0, 4, 0, 5, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0,
    10, 0, 0, 0, 11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 15, 16, 17, 0, 0, 0, 0, 0, 0, 0, 18, 19, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 7, 6, 8, 5, 5, 4, 5, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0
], dtype=np.int32)


class SemKitti(Dataset):
    classes = [
        'car', 'bicycle', 'motorcycle',
        'truck', 'other-vehicle', 'person', 'bicyclist',
        'motorcyclist', 'road', 'parking', 'sidewalk',
        'other-ground', 'building', 'fence', 'vegetation',
        'truck', 'terrain', 'pole', 'traffic-sign'
    ]
    num_classes = len(classes)
    num_per_class = np.array([55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858,
                              240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114,
                              9833174, 129609852, 4506626, 1168181])
    gravity_dim = 2
    class2color = {}
    cmap = [*class2color.values()]

    def __init__(self,
                 dataset_dir: Path,
                 train=True,
                 warmup=False,
                 voxel_max=48000,
                 k=[24, 24, 24, 24],
                 grid_size=[0.08, 0.16, 0.32],
                 visible_sample_stride=0.,
                 alpha=0.,
                 batch_size=8,
                 gs_opts: GaussianOptions = GaussianOptions.default(),
                 ):
        dataset_dir = Path(dataset_dir)
        self.train = train
        self.warmup = warmup
        self.voxel_max = voxel_max
        self.k = k
        self.grid_size = grid_size
        self.visible_sample_stride = visible_sample_stride
        self.alpha = alpha
        self.batch_size = batch_size
        self.gs_opts = gs_opts
        self.class_weights = self.get_class_weights()
        logging.info(f'SemKitti class weights: {self.class_weights}')

        raw_root = dataset_dir / 'sequences'
        self.seq_list = np.sort(os.listdir(raw_root))
        train_list, val_list, _ = get_semantickitti_file_list(raw_root, 'None')
        if train:
            self.data_paths = train_list
        else:
            self.data_paths = val_list
        assert len(self.data_paths) > 0

        if train and warmup:
            self.data_paths = self.data_paths[:8]

        self.cache = {}
        self.cache_size = 20000

    def __len__(self):
        return len(self.data_paths)

    @classmethod
    def get_classes(cls):
        return cls.classes

    def get_class_weights(self):
        weight = self.num_per_class / float(sum(self.num_per_class))
        ce_label_weight = 1 / (weight + 0.02)
        return np.expand_dims(ce_label_weight, axis=0)


    def __getitem__(self, idx):
        if not self.train:
            return self.get_test_item(idx)

        pc_path, label_path = self.data_paths[idx]
        scan = load_scan_kitti(pc_path)
        points = scan[:, 0:3]
        remissions = scan[:, 3]
        labels = load_label_kitti(label_path, remap_lut_read)
        xyz = torch.from_numpy(points).float()
        remissions = torch.from_numpy(remissions).float().unsqueeze(-1)
        lbl = torch.from_numpy(labels).long()

        angle = random.random() * 2 * math.pi
        cos, sin = math.cos(angle), math.sin(angle)
        rotmat = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
        rotmat *= random.uniform(0.8, 1.2)
        xyz = xyz @ rotmat
        xyz += torch.empty_like(xyz).normal_(std=0.005)
        xyz -= xyz.min(dim=0)[0]

        indices = grid_subsampling(xyz, 0.02, 2.5 / 1.5)
        xyz = xyz[indices]
        if xyz.shape[0] > self.voxel_max:
            pt = random.choice(xyz)
            condition = (xyz - pt).square().sum(dim=1).argsort()[:self.voxel_max].sort()[0]  # sort to preserve locality
            xyz = xyz[condition]
            indices = indices[condition]

        remissions = remissions[indices]
        lbl = lbl[indices]
        remissions = remissions.float()

        height = xyz[:, 2:] * 6
        height -= height.min(dim=0, keepdim=True)[0]
        height += torch.empty((1, 1), device=xyz.device).uniform_(-0.1, 0.1) * 6
        feature = torch.cat([xyz, remissions, height], dim=1)

        gs = NaiveGaussian3D(self.gs_opts, batch_size=self.batch_size, device=xyz.device)
        gs.gs_points.__update_attr__('p', xyz)
        gs.gs_points.__update_attr__('f', feature)
        gs.gs_points.__update_attr__('y', lbl)
        gs.projects(xyz, cam_seed=idx, cam_batch=gs.opt.n_cameras*2)
        gs.gs_points = make_gs_points(gs.gs_points, self.k, self.grid_size, None,
                                      up_sample=True, visible_sample_stride=self.visible_sample_stride,
                                      alpha=self.alpha)
        return gs

    def get_test_item(self, idx):
        pick = idx * 5

        pc_path, label_path = self.data_paths[idx]
        scan = load_scan_kitti(pc_path)
        points = scan[:, 0:3]
        remissions = scan[:, 3]
        labels = load_label_kitti(label_path, remap_lut_read)
        xyz = torch.from_numpy(points).float()
        remissions = torch.from_numpy(remissions).float().unsqueeze(-1)
        lbl = torch.from_numpy(labels).long()

        indices = grid_subsampling_test(xyz, 0.02, 2.5 / 1.5, pick=pick)
        xyz = xyz[indices]
        lbl = lbl[indices]
        remissions = remissions[indices].float()

        xyz -= xyz.min(dim=0)[0]
        height = xyz[:, 2:] * 6
        height -= height.min(dim=0, keepdim=True)[0]
        feature = torch.cat([xyz, remissions, height], dim=1)

        gs = NaiveGaussian3D(self.gs_opts, batch_size=self.batch_size, device=xyz.device)
        gs.gs_points.__update_attr__('p', xyz)
        gs.gs_points.__update_attr__('f', feature)
        gs.gs_points.__update_attr__('y', lbl)
        gs.projects(xyz, cam_seed=idx, cam_batch=gs.opt.n_cameras*2)
        gs.gs_points = make_gs_points(gs.gs_points, self.k, self.grid_size, None,
                                      up_sample=True, visible_sample_stride=self.visible_sample_stride,
                                      alpha=self.alpha)
        return gs


def semkitti_collate_fn(batch):
    gs_list = list(batch)
    new_gs = merge_gs_list(gs_list)
    return new_gs
