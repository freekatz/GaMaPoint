import __init__

import torch

from backbone.gs_3d import GaussianOptions
from backbone.mamba_ssm.models import MambaConfig
from utils.config import EasyConfig


class ScanObjectNNConfig(EasyConfig):
    def __init__(self):
        super().__init__()
        self.name = 'ScanObjectNNConfig'
        self.k = [32, 32, 32]
        self.strides = [1, 4, 4]
        self.visible_sample_stride = 0.
        self.num_points = 1024
        gs_opts = GaussianOptions.default()
        gs_opts.n_cameras = 16
        gs_opts.cam_fovy = 120
        self.gs_opts = gs_opts
        self.alpha = gs_opts.n_cameras * 2 / self.num_points
        for s in self.strides:
            self.alpha *= s


class ModelConfig(EasyConfig):
    def __init__(self):
        super().__init__()
        self.name = 'ModelConfig'
        self.train_cfg = ScanObjectNNConfig()
        self.num_classes = 15
        self.bn_momentum = 0.1
        drop_path = 0.1
        stage_cfg = EasyConfig()
        stage_cfg.name = 'StageConfig'
        stage_cfg.in_channels = 1
        stage_cfg.channel_list = [96, 192, 384]
        stage_cfg.head_channels = 2048
        stage_cfg.mamba_blocks = [1, 1, 1]
        stage_cfg.res_blocks = [4, 4, 4]
        stage_cfg.mlp_ratio = 2.
        stage_cfg.bn_momentum = self.bn_momentum
        drop_rates = torch.linspace(0., drop_path, sum(stage_cfg.res_blocks)).split(stage_cfg.res_blocks)
        stage_cfg.drop_paths = [d.tolist() for d in drop_rates]
        stage_cfg.mamba_cfg = MambaConfig.default()
        stage_cfg.hybrid_args = {'hybrid': False}  # whether hybrid mha, {'hybrid': True, 'type': 'post', 'ratio': 0.5}
        stage_cfg.diff_factor = 40.
        stage_cfg.diff_std = [2.2, 4.4, 8.8]
        self.stage_cfg = stage_cfg
