import __init__

import torch

from backbone.gs_3d import GaussianOptions
from backbone.mamba_ssm.models import MambaConfig
from scanobjectnn.configs.config import BaseConfig


class ScanObjectNNConfig(BaseConfig):
    name = 'ScanObjectNNConfig'
    k = [24, 24, 24]
    k_gs = [6, 6, 6]
    strides = [1, 4, 4]
    visible_sample_stride = 0.
    num_points = 1024
    gs_opts = GaussianOptions.default()
    gs_opts.n_cameras = 8
    gs_opts.cam_fovy = 120


class ScanObjectNNWarmupConfig(BaseConfig):
    name = 'ScanObjectNNWarmupConfig'
    k = [24, 24, 24]
    k_gs = [6, 6, 6]
    strides = [1, 4, 4]
    visible_sample_stride = 0.
    num_points = 1024
    gs_opts = GaussianOptions.default()
    gs_opts.n_cameras = 8
    gs_opts.cam_fovy = 120


class GaMaConfig(BaseConfig):
    name = 'GaMaConfig'
    num_classes = 15
    bn_momentum = 0.1
    drop_path = 0.1
    channel_list = [96, 192, 384]
    stage_cfg = BaseConfig()
    stage_cfg.name = 'StageConfig'
    stage_cfg.in_channels = 1
    stage_cfg.channel_list = channel_list
    stage_cfg.head_channels = 768
    stage_cfg.mamba_blocks = [1, 1, 1]
    stage_cfg.res_blocks = [4, 4, 4]
    stage_cfg.mlp_ratio = 2.
    stage_cfg.bn_momentum = bn_momentum
    drop_rates = torch.linspace(0., drop_path, sum(stage_cfg.res_blocks)).split(stage_cfg.res_blocks)
    stage_cfg.drop_paths = [d.tolist() for d in drop_rates]
    stage_cfg.mamba_cfg = MambaConfig.default()
    stage_cfg.hybrid_args = {'hybrid': False}  # whether hybrid mha, {'hybrid': True, 'type': 'post', 'ratio': 0.5}
    stage_cfg.diff_factor = 40.
    stage_cfg.diff_std = [2.2, 4.4, 8.8]
