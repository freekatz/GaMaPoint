import logging
import math

import torch
import torch.nn as nn
from timm.models.layers import DropPath

from backbone.gs_3d import NaiveGaussian3D
from backbone.mamba_ssm.custom import StructuredMask
from backbone.mamba_ssm.custom.order import Order
from backbone.mamba_ssm.models import MambaConfig, MixerModel
from utils.cutils import knn_edge_maxpooling


class LocalAggregation(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bn_momentum: float = 0.1,
                 init_weight: float = 0.,
                 **kwargs
                 ):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels, momentum=bn_momentum)
        nn.init.constant_(self.bn.weight, init_weight)

    def forward(self, f, group_idx):
        """
        :param f: [B, N, channels]
        :param group_idx: [B, N, K]
        :return: [B, N, channels]
        """
        assert len(f.shape) == 3
        B, N, C = f.shape
        f = self.proj(f)
        f = knn_edge_maxpooling(f, group_idx, self.training)
        f = self.bn(f.view(B * N, -1)).view(B, N, -1)
        return f


class Mlp(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 bn_momentum,
                 init_weight=0.,
                 **kwargs
                 ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, in_channels, bias=False),
            nn.BatchNorm1d(in_channels, momentum=bn_momentum),
        )
        nn.init.constant_(self.mlp[-1].weight, init_weight)

    def forward(self, f):
        """
        :param f: [B, N, in_channels]
        :return: [B, N, in_channels]
        """
        assert len(f.shape) == 3
        B, N, C = f.shape
        f = self.mlp(f.view(B * N, -1)).view(B, N, -1)
        return f


class InvResMLP(nn.Module):
    def __init__(self,
                 channels,
                 res_blocks,
                 mlp_ratio,
                 bn_momentum,
                 drop_path,
                 **kwargs
                 ):
        super().__init__()
        self.res_blocks = res_blocks

        hidden_channels = round(channels * mlp_ratio)
        self.mlp = Mlp(channels, hidden_channels, bn_momentum, 0.2)
        self.blocks = nn.ModuleList([
            LocalAggregation(
                channels,
                channels,
                bn_momentum,
            )
            for _ in range(res_blocks)
        ])
        self.mlps = nn.ModuleList([
            Mlp(
                channels,
                hidden_channels,
                bn_momentum,
            )
            for _ in range(res_blocks // 2)
        ])
        self.drop_paths = nn.ModuleList([
            DropPath(d) for d in drop_path
        ])
        self.use_drop = [d > 0. for d in drop_path]

    def drop_path(self, f, block_index, pts):
        if not self.use_drop[block_index] or not self.training:
            return f
        return torch.cat([self.drop_paths[block_index](x) for x in torch.split(f, pts, dim=1)], dim=1)

    def forward(self, f, group_idx, pts):
        """
        :param f: [B, N, channels]
        :param group_idx: [B, N, K]
        :return: [B, N, channels]
        """
        assert len(f.shape) == 3
        assert len(group_idx.shape) == 3
        assert pts is not None

        f = f + self.drop_path(self.mlp(f), 0, pts)
        for i in range(self.res_blocks):
            f = f + self.drop_path(self.blocks[i](f, group_idx), i, pts)
            if i % 2 == 1:
                f = f + self.drop_path(self.mlps[i // 2](f), i, pts)
        return f


def make_hybrid_idx(n_layer, hybrid_type, ratio) -> list:
    assert hybrid_type in ['pre', 'post']
    n_hybrid = int(n_layer * ratio)
    if n_hybrid == 0 and ratio > 0:
        n_hybrid = 1

    attn_idx_left, attn_idx_right = 0, 0
    if hybrid_type == 'post':
        attn_idx_left, attn_idx_right = n_layer - n_hybrid, n_layer
    elif hybrid_type == 'pre':
        attn_idx_left, attn_idx_right = 0, n_hybrid
    return [i for i in range(attn_idx_left, attn_idx_right)]


def self_adapt_heads(d_model: int) -> int:
    target_headdim = 64
    num_heads = math.ceil(d_model / target_headdim)
    num_heads |= num_heads >> 1
    num_heads |= num_heads >> 2
    num_heads |= num_heads >> 4
    num_heads |= num_heads >> 8
    num_heads |= num_heads >> 16
    num_heads += 1
    return max(4, num_heads)


def create_mixer(
        config: MambaConfig,
        d_model: int,
        hybrid_args: dict,
):
    config.d_model = d_model
    n_layer = config.n_layer
    d_intermediate = config.d_intermediate
    ssm_cfg = config.ssm_cfg
    attn_cfg = config.attn_cfg
    rms_norm = config.get('rms_norm', True)
    residual_in_fp32 = config.get('residual_in_fp32', True)

    hybrid = hybrid_args.get('hybrid', False)
    hybrid_type = hybrid_args.get('type')
    hybrid_ratio = hybrid_args.get('ratio', 0.)

    if attn_cfg.get('self_adapt_heads', False):
        num_heads = self_adapt_heads(d_model)
    else:
        num_heads = attn_cfg.get('num_heads', self_adapt_heads(d_model))

    expand = ssm_cfg.get('expand', 2)
    ssm_cfg.d_state = min(d_model * expand, 1024)
    ssm_cfg.headdim = min(d_model // num_heads, 192)
    attn_layer_idx = [] if not hybrid else make_hybrid_idx(n_layer, hybrid_type, hybrid_ratio)

    min_heads = max(d_model // ssm_cfg.headdim, 8)
    if num_heads < min_heads:
        logging.warning(
            f'num heads {num_heads} < min heads {min_heads}, will replace with {min_heads}, d_model={d_model}')
        num_heads = min_heads
    attn_cfg.num_heads = num_heads
    attn_cfg = attn_cfg if hybrid else None
    return MixerModel(
        d_model=d_model,
        n_layer=n_layer,
        d_intermediate=d_intermediate,
        ssm_cfg=ssm_cfg,
        attn_layer_idx=attn_layer_idx,
        attn_cfg=attn_cfg,
        rms_norm=rms_norm,
        residual_in_fp32=residual_in_fp32,
    )


class PointMambaLayer(nn.Module):
    def __init__(self,
                 layer_index: int,
                 channels: int,
                 config: MambaConfig,
                 hybrid_args: dict,
                 bn_momentum: float,
                 **kwargs,
                 ):
        super().__init__()
        self.layer_index = layer_index
        self.config = config

        self.mixer = create_mixer(config, channels, hybrid_args)
        self.alpha = nn.Parameter(torch.tensor([0.5], dtype=torch.float32) * 100)
        self.bn = nn.BatchNorm1d(channels, momentum=bn_momentum)

    def forward(self, p, p_gs, f, gs: NaiveGaussian3D):
        assert len(f.shape) == 2
        f = f.unsqueeze(0)
        B, N, C = f.shape
        mask = None
        f_global = self.mixer(input_ids=f,
                              mask=mask, gs=None, order=None)
        alpha = self.alpha.sigmoid()
        f = f_global * alpha + f * (1 - alpha)
        f = self.bn(f.view(B * N, -1)).view(B, N, -1)
        return f.squeeze(0)
