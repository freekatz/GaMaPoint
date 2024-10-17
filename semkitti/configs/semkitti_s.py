import __init__

import torch

from backbone.gs_3d import GaussianOptions
from backbone.mamba_ssm.models import MambaConfig
from utils.config import EasyConfig


class SemKittiConfig(EasyConfig):
    def __init__(self):
        super().__init__()
        self.name = 'SemKittiConfig'
        self.k = [16, 16, 16, 16, 16]
        self.grid_size = [0.04, 0.08, 0.16, 0.32]
        self.voxel_max = 48000
        gs_opts = GaussianOptions.default()
        gs_opts.n_cameras = 64
        gs_opts.cam_fovy = 90
        self.gs_opts = gs_opts
        self.alpha = 0.1


class SemKittiWarmupConfig(EasyConfig):
    def __init__(self):
        super().__init__()
        self.name = 'SemKittiWarmupConfig'
        self.k = [16, 16, 16, 16, 16]
        self.grid_size = [0.04, 0.08, 0.16, 0.32]
        self.voxel_max = 48000
        gs_opts = GaussianOptions.default()
        gs_opts.n_cameras = 64
        gs_opts.cam_fovy = 90
        self.gs_opts = gs_opts
        self.alpha = 0.1


class ModelConfig(EasyConfig):
    def __init__(self):
        super().__init__()
        self.name = 'ModelConfig'
        self.train_cfg = SemKittiConfig()
        self.warmup_cfg = SemKittiWarmupConfig()
        self.num_classes = 19
        self.bn_momentum = 0.02
        drop_path = 0.1
        backbone_cfg = EasyConfig()
        backbone_cfg.name = 'BackboneConfig'
        backbone_cfg.in_channels = 5
        backbone_cfg.channel_list = [64, 96, 160, 288, 512]
        backbone_cfg.head_channels = 288
        backbone_cfg.mamba_blocks = [1, 1, 1, 1, 1]
        backbone_cfg.res_blocks = [4, 4, 4, 8, 4]
        backbone_cfg.mlp_ratio = 2.
        backbone_cfg.bn_momentum = self.bn_momentum
        drop_rates = torch.linspace(0., drop_path, sum(backbone_cfg.res_blocks)).split(backbone_cfg.res_blocks)
        backbone_cfg.drop_paths = [d.tolist() for d in drop_rates]
        backbone_cfg.head_drops = torch.linspace(0., 0.15, len(backbone_cfg.res_blocks)).tolist()
        backbone_cfg.mamba_cfg = MambaConfig.default()
        backbone_cfg.hybrid_args = {'hybrid': False}  # whether hybrid mha, {'hybrid': True, 'type': 'post', 'ratio': 0.5}
        backbone_cfg.gs_opts = self.train_cfg.gs_opts
        backbone_cfg.diff_factor = 60.
        backbone_cfg.diff_std = [4, 8, 12, 24, 40]
        # backbone_cfg.diff_std = None
        self.backbone_cfg = backbone_cfg
