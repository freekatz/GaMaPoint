import torch


def points_centroid(xyz):
    """
    :param xyz: [B, N, 3]
    :return: [B, 3]
    """
    return torch.mean(xyz, dim=1)


def points_scaler(xyz, scale=2.):
    """
    :param xyz: []
    :param scale: float, scale factor, by default 2.0, which means scale into [-0.5*2, 0.5*2]
    :return: []
    """
    xyz = (xyz - xyz.min()) / (xyz.max() - xyz.min()) - 0.5
    return xyz * scale

