import __init__

import torch

from backbone.gs_3d import GaussianOptions
from backbone.mamba_ssm.models import MambaConfig
from utils.config import EasyConfig


class ScanNetV2Config(EasyConfig):
    def __init__(self):
        super().__init__()
        self.name = 'ScanNetV2Config'
        self.k = [24, 24, 24, 24, 24]
        self.use_gs = False
        self.k_gs = [6, 6, 6, 6, 6]
        self.grid_size = [0.04, 0.08, 0.16, 0.32]
        self.voxel_max = 80000
        gs_opts = GaussianOptions.default()
        gs_opts.n_cameras = 16
        gs_opts.cam_fovy = 120
        self.gs_opts = gs_opts
        self.alpha = min(1.0, gs_opts.n_cameras * 2 / self.voxel_max * 4 ** len(self.grid_size))


class ScanNetV2WarmupConfig(EasyConfig):
    def __init__(self):
        super().__init__()
        self.name = 'ScanNetV2WarmupConfig'
        self.k = [24, 24, 24, 24, 24]
        self.use_gs = False
        self.k_gs = [6, 6, 6, 6, 6]
        self.grid_size = [0.04, 0.08, 0.16, 0.32]
        self.voxel_max = 80000
        gs_opts = GaussianOptions.default()
        gs_opts.n_cameras = 16
        gs_opts.cam_fovy = 120
        self.gs_opts = gs_opts
        self.alpha = min(1.0, gs_opts.n_cameras * 2 / self.voxel_max * 4 ** len(self.grid_size))


class ModelConfig(EasyConfig):
    def __init__(self):
        super().__init__()
        self.name = 'ModelConfig'
        self.train_cfg = ScanNetV2Config()
        self.warmup_cfg = ScanNetV2WarmupConfig()
        self.num_classes = 20
        self.bn_momentum = 0.02
        drop_path = 0.1
        stage_cfg = EasyConfig()
        stage_cfg.name = 'StageConfig'
        stage_cfg.in_channels = 7
        stage_cfg.channel_list = [64, 96, 160, 288, 512]
        stage_cfg.head_channels = 288
        stage_cfg.mamba_blocks = [1, 1, 1, 1, 1]
        stage_cfg.res_blocks = [4, 4, 4, 8, 4]
        stage_cfg.mlp_ratio = 2.
        stage_cfg.beta = self.train_cfg.alpha
        stage_cfg.use_gs = self.train_cfg.use_gs
        stage_cfg.bn_momentum = self.bn_momentum
        drop_rates = torch.linspace(0., drop_path, sum(stage_cfg.res_blocks)).split(stage_cfg.res_blocks)
        stage_cfg.drop_paths = [d.tolist() for d in drop_rates]
        stage_cfg.head_drops = torch.linspace(0., 0.2, len(stage_cfg.res_blocks)).tolist()
        stage_cfg.mamba_cfg = MambaConfig.default()
        stage_cfg.hybrid_args = {'hybrid': False}  # whether hybrid mha, {'hybrid': True, 'type': 'post', 'ratio': 0.5}
        stage_cfg.diff_factor = 60.
        stage_cfg.diff_std = [1.6, 2.5, 5, 10, 20]
        self.stage_cfg = stage_cfg
