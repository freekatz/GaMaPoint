import torch

import __init__

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
    channel_list = [64, 192, 384, 768]
    encoder_cfg = BaseConfig()
    encoder_cfg.name = 'EncoderConfig'
    encoder_cfg.in_channels = 4
    encoder_cfg.channel_list = channel_list
    encoder_cfg.mamba_blocks = [1, 1, 1, 1]
    encoder_cfg.res_blocks = [2, 2, 4, 2]
    encoder_cfg.mlp_ratio = 2.
    encoder_cfg.bn_momentum = bn_momentum
    drop_rates = torch.linspace(0., drop_path, sum(encoder_cfg.res_blocks)).split(encoder_cfg.res_blocks)
    encoder_cfg.drop_paths = [d.tolist() for d in drop_rates]
    encoder_cfg.mamba_cfg = MambaConfig.default()
    encoder_cfg.hybrid_args = {'hybrid': False, 'type': 'post', 'ratio': 0.5}

    decoder_cfg = BaseConfig()
    decoder_cfg.name = 'DecoderConfig'
    decoder_cfg.channel_list = channel_list
    decoder_cfg.out_channels = 256
    decoder_cfg.bn_momentum = bn_momentum
