import __init__

import logging
import math

import torch
from torch import nn

from backbone.gs_3d import NaiveGaussian3D
from backbone.mamba_ssm.custom import StructuredMask
from backbone.mamba_ssm.custom.order import Order
from backbone.mamba_ssm.models import MambaConfig, MixerModel
from utils.cutils import knn_edge_maxpooling


class SpatialEmbedding(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 bn_momentum,
                 **kwargs
                 ):
        super().__init__()
        assert len(hidden_channels) == 3
        self.embed = nn.Sequential(
            nn.Linear(in_channels, hidden_channels[0], bias=False),
            nn.BatchNorm1d(hidden_channels[0], momentum=bn_momentum),
            nn.GELU(),
            nn.Linear(hidden_channels[0], hidden_channels[1], bias=False),
            nn.BatchNorm1d(hidden_channels[1], momentum=bn_momentum),
            nn.GELU(),
            nn.Linear(hidden_channels[1], hidden_channels[2], bias=False),
        )
        self.proj = nn.Identity() \
            if hidden_channels[2] == out_channels \
            else nn.Linear(hidden_channels[2], out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels, momentum=bn_momentum)
        nn.init.constant_(self.bn.weight, 0.5)

    def forward(self, f, group_idx):
        assert len(f.shape) == 2
        N, K = group_idx.shape
        f = self.embed(f).view(N, K, -1)
        f = f.max(dim=1)[0]
        f = self.proj(f)
        f = self.bn(f)
        return f


class SetAbstraction(nn.Module):
    def __init__(self,
                 layer_index,
                 in_channels,
                 out_channels,
                 bn_momentum,
                 **kwargs
                 ):
        super().__init__()
        self.layer_index = layer_index
        is_head = self.layer_index == 0
        self.is_head = is_head
        self.in_channels = in_channels

        if not is_head:
            self.skip_proj = nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                nn.BatchNorm1d(out_channels, momentum=bn_momentum)
            )
            self.la = LocalAggregation(in_channels, out_channels, bn_momentum, 0.3)
            nn.init.constant_(self.skip_proj[1].weight, 0.3)

        nbr_in_channels = 3+in_channels if is_head else 3
        hidden_channels = 32 if is_head else 16
        nbr_hidden_channels = [hidden_channels, hidden_channels//2, out_channels if is_head else 32]
        nbr_out_channels = out_channels
        self.spe = SpatialEmbedding(
            nbr_in_channels,
            nbr_hidden_channels,
            nbr_out_channels,
            bn_momentum,
        )
        self.spe_gs = SpatialEmbedding(
            nbr_in_channels,
            nbr_hidden_channels,
            nbr_out_channels,
            bn_momentum,
        )
        self.alpha = nn.Parameter(torch.ones((1,), dtype=torch.float32) * 100)

    def forward(self, p, f, gs: NaiveGaussian3D):
        assert len(f.shape) == 2
        if not self.is_head:
            idx = gs.gs_points.idx_ds[self.layer_index-1]
            pre_group_idx = gs.gs_points.idx_group[self.layer_index-1]
            f = self.skip_proj(f)[idx] + self.la(f.unsqueeze(0), pre_group_idx.unsqueeze(0)).squeeze(0)[idx]

        p_nbr, group_idx = gs.gs_points.grouping('p', self.layer_index, need_idx=True)
        p_nbr = p_nbr - p.unsqueeze(1)
        if self.is_head:
            f_nbr = f[group_idx]
            f_nbr = torch.cat([p_nbr, f_nbr], dim=-1).view(-1, 3 + self.in_channels)
        else:
            f_nbr = p_nbr.view(-1, 3)

        # gs_group_idx = gs.gs_points.idx_gs_group[self.layer_index]
        f_nbr = self.spe(f_nbr, group_idx)
        # f_gs_nbr = self.spe_gs(f_nbr, gs_group_idx)

        alpha = self.alpha.sigmoid()
        if self.is_head:
            # f = f_nbr * alpha + f_gs_nbr * (1-alpha)
            f = f_nbr
        else:
            # f = f + f_nbr * alpha + f_gs_nbr * (1-alpha)
            f = f + f_nbr

        return f


class Mlp(nn.Module):
    def __init__(self,
                 in_channels,
                 mlp_ratio,
                 bn_momentum,
                 init_weight=0.,
                 **kwargs
                 ):
        super().__init__()
        hidden_channels = round(in_channels * mlp_ratio)

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
        :param f: [N, channels]
        :param group_idx: [N, K]
        :return: [N, channels]
        """
        assert len(f.shape) == 3
        B, N, C = f.shape
        x = self.proj(f)
        x = knn_edge_maxpooling(x.contiguous(), group_idx.contiguous(), self.training)
        x = self.bn(x.view(B * N, -1)).view(B, N, -1)
        return x


class InvResMLP(nn.Module):
    def __init__(self,
                 channels,
                 res_blocks,
                 mlp_ratio,
                 bn_momentum,
                 **kwargs
                 ):
        super().__init__()
        self.res_blocks = res_blocks

        self.mlp = Mlp(channels, mlp_ratio, bn_momentum, 0.2)
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
                mlp_ratio,
                bn_momentum,
            )
            for _ in range(res_blocks // 2)
        ])

    def forward(self, f, group_idx):
        """
        :param f: [N, channels]
        :param group_idx: [N, K]
        :return: [N, channels]
        """
        assert len(f.shape) == 2
        f = f.unsqueeze(0)
        group_idx = group_idx.unsqueeze(0)
        f = f + self.mlp(f)
        for i in range(self.res_blocks):
            f = f + self.blocks[i](f, group_idx)
            if i % 2 == 1:
                f = f + self.mlps[i//2](f)
        return f.squeeze(0)


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
    fused_add_norm = config.get('fused_add_norm', True)

    hybrid = hybrid_args.get('hybrid', False)
    hybrid_type = hybrid_args.get('type')
    hybrid_ratio = hybrid_args.get('ratio', 0.)

    if attn_cfg.get('self_adapt_heads', False):
        num_heads = self_adapt_heads(d_model)
    else:
        num_heads = attn_cfg.get('num_heads', self_adapt_heads(d_model))

    expand = ssm_cfg.get('expand', 2)
    ssm_cfg.d_state = min(d_model * expand, 1024)
    ssm_cfg.headdim = min(d_model // num_heads, 128)
    attn_layer_idx = [] if not hybrid else make_hybrid_idx(n_layer, hybrid_type, hybrid_ratio)

    if d_model // ssm_cfg.headdim > num_heads:
        logging.warning(
            f'num heads {d_model // ssm_cfg.headdim} > {num_heads}, will replace with {d_model // ssm_cfg.headdim}')
        num_heads = d_model // ssm_cfg.headdim
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
        fused_add_norm=fused_add_norm,
    )


class PointMambaLayer(nn.Module):
    def __init__(self,
                 layer_index: int,
                 channels: int,
                 config: MambaConfig,
                 hybrid_args: dict,
                 **kwargs,
                 ):
        super().__init__()
        self.layer_index = layer_index
        self.config = config

        self.mixer = create_mixer(config, channels, hybrid_args)
        self.alpha = nn.Parameter(torch.tensor([0.5], dtype=torch.float32) * 100)

    def forward(self, p, p_gs, f, gs: NaiveGaussian3D):
        assert len(f.shape) == 2
        cov3d = gs.gs_points.cov3d

        # get order
        cam_order = p_gs[:, 2]
        idx = torch.argsort(cam_order, dim=0, descending=True)
        order = Order(idx.unsqueeze(0))
        # apply mask
        mask = None
        if self.config.use_mask:
            mask = StructuredMask(
                mask_type='cov3d',
                mask_params={'cov3d': cov3d, 'd_model': self.config.d_model}
            )

        # todo
        mask = None
        f_global = self.mixer(input_ids=f.unsqueeze(0),
                              mask=mask, gs=gs, order=order)
        alpha = self.alpha.sigmoid()
        f = f_global.squeeze(0) * alpha + f * (1 - alpha)
        return f

