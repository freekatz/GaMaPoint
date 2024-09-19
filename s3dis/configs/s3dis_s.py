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
