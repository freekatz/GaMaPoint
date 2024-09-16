import __init__

from types import SimpleNamespace

from backbone.ops.gs_3d import GaussianOptions


class S3disConfig(SimpleNamespace):
    k = [24, 32, 32, 32]
    grid_size = [0.04, 0.08, 0.16, 0.32]
    voxel_max = 30000
    gs_opts = GaussianOptions.default()


class S3disWarmupConfig(SimpleNamespace):
    k = [24, 24, 24, 24]
    grid_size = [0.04, 3.5, 3.5, 3.5]
    voxel_max = 30000
    gs_opts = GaussianOptions.default()


class GaMaConfig(SimpleNamespace):
    num_classes = 13

