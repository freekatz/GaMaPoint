import __init__

import torch

from backbone.gs_3d import GaussianOptions
from backbone.mamba_ssm.models import MambaConfig
from s3dis.configs.config import BaseConfig


class S3disConfig(BaseConfig):
    name = 'S3disConfig'
    k = [24, 24, 24, 24]
    grid_size = [0.04, 0.08, 0.16, 0.32]
    voxel_max = 30000
    gs_opts = GaussianOptions.default()


class S3disWarmupConfig(BaseConfig):
    name = 'S3disWarmupConfig'
    k = [24, 24, 24, 24]
    grid_size = [0.04, 3.5, 3.5, 3.5]
    voxel_max = 30000
    gs_opts = GaussianOptions.default()


class GaMaConfig(BaseConfig):
    name = 'GaMaConfig'
    num_classes = 13
    bn_momentum = 0.02
    drop_path = 0.1
    channel_list = [64, 128, 256, 512]
    stage_cfg = BaseConfig()
    stage_cfg.name = 'EncoderConfig'
    stage_cfg.in_channels = 4
    stage_cfg.channel_list = channel_list
    stage_cfg.head_channels = 256
    stage_cfg.mamba_blocks = [1, 1, 1, 1]
    stage_cfg.res_blocks = [4, 4, 8, 4]
    stage_cfg.mlp_ratio = 2.
    stage_cfg.bn_momentum = bn_momentum
    drop_rates = torch.linspace(0., drop_path, sum(stage_cfg.res_blocks)).split(stage_cfg.res_blocks)
    stage_cfg.drop_paths = [d.tolist() for d in drop_rates]
    stage_cfg.head_drops = torch.linspace(0., 0.15, len(stage_cfg.res_blocks)).tolist()
    stage_cfg.mamba_cfg = MambaConfig.default()
    stage_cfg.hybrid_args = {'hybrid': False}  # whether hybrid mha, {'hybrid': True, 'type': 'post', 'ratio': 0.5}

