import __init__

import torch

from backbone.gs_3d import GaussianOptions
from backbone.mamba_ssm.models import MambaConfig
from utils.config import EasyConfig


class ModelNet40Config(EasyConfig):
    def __init__(self):
        super().__init__()
        self.name = 'ModelNet40Config'
        self.k = [32, 32, 32]
        self.n_samples = [2048, 512, 128]
        self.visible_sample_stride = 0.
        self.num_points = 2048
        gs_opts = GaussianOptions.default()
        gs_opts.n_cameras = 8
        gs_opts.cam_fovy = 120
        self.gs_opts = gs_opts
        self.alpha = 0.1


class ModelConfig(EasyConfig):
    def __init__(self):
        super().__init__()
        self.name = 'ModelConfig'
        self.train_cfg = ModelNet40Config()
        self.num_classes = 40
        self.bn_momentum = 0.1
        drop_path = 0.15
        backbone_cfg = EasyConfig()
        backbone_cfg.name = 'BackboneConfig'
        backbone_cfg.in_channels = 4
        backbone_cfg.channel_list = [96, 192, 384]
        backbone_cfg.head_channels = 2048
        backbone_cfg.mamba_blocks = [1, 1, 1]
        backbone_cfg.res_blocks = [4, 4, 4]
        backbone_cfg.mlp_ratio = 2.
        backbone_cfg.bn_momentum = self.bn_momentum
        drop_rates = torch.linspace(0., drop_path, sum(backbone_cfg.res_blocks)).split(backbone_cfg.res_blocks)
        backbone_cfg.drop_paths = [d.tolist() for d in drop_rates]
        backbone_cfg.mamba_cfg = MambaConfig.default()
        backbone_cfg.hybrid_args = {'hybrid': False}  # whether hybrid mha, {'hybrid': True, 'type': 'post', 'ratio': 0.5}
        backbone_cfg.gs_opts = self.train_cfg.gs_opts
        backbone_cfg.diff_factor = 40.
        backbone_cfg.diff_std = [2.8, 5.3, 10]
        # backbone_cfg.diff_std = None
        self.backbone_cfg = backbone_cfg
