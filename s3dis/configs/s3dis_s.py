import __init__

from types import SimpleNamespace

from backbone.gs_3d import GaussianOptions
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
    bn_momentum = 0.
    stage_cfg = BaseConfig()
    stage_cfg.name = 'StageConfig'
    stage_cfg.in_channels = 4
    stage_cfg.channel_list = [64, 128, 256, 512]
    stage_cfg.mamba_blocks = [1, 1, 2, 1]
    stage_cfg.res_blocks = [4, 4, 8, 4]
    stage_cfg.mlp_ratio = 2.
    stage_cfg.bn_momentum = bn_momentum
    stage_cfg.hybrid_args = {'hybrid': False, 'type': 'post', 'ratio': 0.5}
    stage_cfg.channel_list = [64, 128, 256, 512]
    stage_cfg.out_channels = 256
    stage_cfg.bn_momentum = bn_momentum
    stage_cfg.task_type = 'seg'

