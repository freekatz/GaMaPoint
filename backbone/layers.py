import logging
import math

import torch
import torch.nn as nn
from timm.models.layers import DropPath

from backbone.gs_3d import NaiveGaussian3D, GaussianOptions
from backbone.mamba_ssm.custom.order import Order
from backbone.mamba_ssm.models import MambaConfig, MixerModel
from utils.cutils import knn_edge_maxpooling
from utils.model_utils import checkpoint


class SetAbstraction(nn.Module):
    def __init__(self,
                 layer_index=0,
                 in_channels=4,
                 channel_list=[64, 128, 256, 512],
                 bn_momentum=0.02,
                 use_cp=False,
                 ):
        super().__init__()
        self.use_cp = use_cp
        self.layer_index = layer_index
        is_head = self.layer_index == 0
        self.is_head = is_head
        self.in_channels = in_channels if is_head else channel_list[layer_index - 1]
        self.out_channels = channel_list[layer_index]

        embed_in_channels = 3 + self.in_channels if is_head else 3
        embed_hidden_channels = channel_list[0] // 2 if is_head else channel_list[0] // 4
        embed_out_channels = self.out_channels if is_head else channel_list[0] // 2

        self.embed = nn.Sequential(
            nn.Linear(embed_in_channels, embed_hidden_channels // 2, bias=False),
            nn.BatchNorm1d(embed_hidden_channels // 2, momentum=bn_momentum),
            nn.GELU(),
            nn.Linear(embed_hidden_channels // 2, embed_hidden_channels, bias=False),
            nn.BatchNorm1d(embed_hidden_channels, momentum=bn_momentum),
            nn.GELU(),
            nn.Linear(embed_hidden_channels, embed_out_channels, bias=False),
        )
        self.proj = nn.Identity() if is_head else nn.Linear(embed_out_channels, self.out_channels, bias=False)
        self.bn = nn.BatchNorm1d(self.out_channels, momentum=bn_momentum)
        nn.init.constant_(self.bn.weight, 0.8 if is_head else 0.2)

    def forward(self, p, f, group_idx):
        assert len(f.shape) == 2

        p_group = p[group_idx] - p.unsqueeze(1)
        if self.is_head:
            f_group = f[group_idx]
            f_group = torch.cat([p_group, f_group], dim=-1).view(-1, 3 + self.in_channels)
        else:
            f_group = p_group.view(-1, 3)

        N, K = group_idx.shape
        embed_fn = lambda x: self.embed(x).view(N, K, -1).max(dim=1)[0]
        f_group = embed_fn(f_group) if not self.use_cp \
            else checkpoint(embed_fn, f_group)
        f_group = self.proj(f_group)
        f_group = self.bn(f_group)

        f = f_group if self.is_head else f_group + f
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
    min_heads = 8
    min_head_dims = 64
    num_heads = math.ceil(d_model / min_head_dims)
    num_heads |= num_heads >> 1
    num_heads |= num_heads >> 2
    num_heads |= num_heads >> 4
    num_heads |= num_heads >> 8
    num_heads |= num_heads >> 16
    num_heads += 1
    return max(min_heads, num_heads)


def create_mixer(
        config: MambaConfig,
        d_model: int,
        hybrid_args: dict,
):
    config.d_model = d_model
    n_layer = config.n_layer
    ssm_cfg = config.ssm_cfg
    attn_cfg = config.attn_cfg
    d_intermediate = config.get('d_intermediate', 0)
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
                 gs_opts: GaussianOptions,
                 **kwargs,
                 ):
        super().__init__()
        self.layer_index = layer_index
        self.config = config
        if self.config.use_pos:
            self.pos_embed = nn.Parameter(torch.randn([gs_opts.n_cameras * 2, channels], dtype=torch.float32))
        self.mixer = create_mixer(config, channels, hybrid_args)
        self.bn = nn.BatchNorm1d(channels, momentum=bn_momentum)

    def forward(self, f, f_gs, gs: NaiveGaussian3D):
        assert len(f.shape) == 2
        if self.config.use_pos:
            pos_embed = f_gs @ self.pos_embed  # [N, M] @ [M, C] = [N, C]
            pos_embed = pos_embed.unsqueeze(0)
        else:
            pos_embed = None
        f = f.unsqueeze(0)  # [1, N, C]
        B, N, C = f.shape
        f = f + self.mixer(input_ids=f, pos_embed=pos_embed, mask=None, gs=gs, order=None)
        f = self.bn(f.view(B * N, -1)).view(B, N, -1)
        return f.squeeze(0)
