import __init__

import torch

from backbone.gs_3d import GaussianOptions
from backbone.mamba_ssm.models import MambaConfig
from utils.config import EasyConfig


class S3disConfig(EasyConfig):
    def __init__(self):
        super().__init__()
        self.name = 'S3disConfig'
        self.k = [24, 24, 24, 24]
        self.k_gs = [6, 6, 6, 6]
        self.grid_size = [0.08, 0.16, 0.32]
        self.voxel_max = 30000
        gs_opts = GaussianOptions.default()
        gs_opts.n_cameras = 48
        gs_opts.cam_fovy = 120
        self.gs_opts = gs_opts
        self.alpha = min(1.0, gs_opts.n_cameras * 2 / self.voxel_max * 4 ** len(self.grid_size))


class S3disWarmupConfig(EasyConfig):
    def __init__(self):
        super().__init__()
        self.name = 'S3disWarmupConfig'
        self.k = [24, 24, 24, 24]
        self.k_gs = [6, 6, 6, 6]
        self.grid_size = [0.08, 0.16, 0.32]
        self.voxel_max = 30000
        gs_opts = GaussianOptions.default()
        gs_opts.n_cameras = 48
        gs_opts.cam_fovy = 120
        self.gs_opts = gs_opts
        self.alpha = min(1.0, gs_opts.n_cameras * 2 / self.voxel_max * 4 ** len(self.grid_size))


class ModelConfig(EasyConfig):
    def __init__(self):
        super().__init__()
        self.name = 'ModelConfig'
        self.train_cfg = S3disConfig()
        self.warmup_cfg = S3disWarmupConfig()
        self.num_classes = 13
        self.bn_momentum = 0.02
        drop_path = 0.1
        stage_cfg = EasyConfig()
        stage_cfg.name = 'StageConfig'
        stage_cfg.in_channels = 4
        stage_cfg.channel_list = [64, 128, 256, 512]
        stage_cfg.head_channels = 256
        stage_cfg.mamba_blocks = [1, 1, 1, 1]
        stage_cfg.res_blocks = [4, 4, 8, 4]
        stage_cfg.mlp_ratio = 2.
        stage_cfg.beta = self.train_cfg.alpha
        stage_cfg.bn_momentum = self.bn_momentum
        drop_rates = torch.linspace(0., drop_path, sum(stage_cfg.res_blocks)).split(stage_cfg.res_blocks)
        stage_cfg.drop_paths = [d.tolist() for d in drop_rates]
        stage_cfg.head_drops = torch.linspace(0., 0.15, len(stage_cfg.res_blocks)).tolist()
        stage_cfg.mamba_cfg = MambaConfig.default()
        stage_cfg.hybrid_args = {'hybrid': False}  # whether hybrid mha, {'hybrid': True, 'type': 'post', 'ratio': 0.5}
        stage_cfg.diff_factor = 40.
        stage_cfg.diff_std = [1.6, 3.2, 6.4, 12.8]
        self.stage_cfg = stage_cfg
