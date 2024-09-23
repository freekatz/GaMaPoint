import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from timm.models.layers import DropPath

from backbone.gs_3d import NaiveGaussian3D
from backbone.mamba_ssm.custom import StructuredMask
from backbone.mamba_ssm.custom.order import Order
from backbone.mamba_ssm.models import MambaConfig, MixerModel
from utils.cutils import knn_edge_maxpooling


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
            f'num heads {num_heads} < min heads {min_heads}, will replace with {min_heads}')
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
        f_global = self.mixer(input_ids=f,
                              mask=mask, gs=gs, order=order)
        alpha = self.alpha.sigmoid()
        f = f_global * alpha + f * (1 - alpha)
        f = self.bn(f.view(B * N, -1)).view(B, N, -1)
        return f.squeeze(0)


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
        :param f: [N, channels]
        :param group_idx: [N, K]
        :return: [N, channels]
        """
        assert len(f.shape) == 2
        assert pts is not None

        f = f.unsqueeze(0)
        group_idx = group_idx.unsqueeze(0)
        f = f + self.drop_path(self.mlp(f), 0, pts)
        for i in range(self.res_blocks):
            f = f + self.drop_path(self.blocks[i](f, group_idx), i, pts)
            if i % 2 == 1:
                f = f + self.drop_path(self.mlps[i // 2](f), i, pts)
        return f.squeeze(0)


class Stage(nn.Module):
    def __init__(self,
                 layer_index=0,
                 in_channels=4,
                 channel_list=[64, 128, 256, 512],
                 head_channels=256,
                 mamba_blocks=[1, 1, 1, 1],
                 res_blocks=[4, 4, 8, 4],
                 mlp_ratio=2.,
                 bn_momentum=0.02,
                 drop_paths=None,
                 head_drops=None,
                 mamba_config=MambaConfig().default(),
                 hybrid_args={'hybrid': False},
                 cross_std=None,
                 **kwargs
                 ):
        super().__init__()
        assert cross_std is not None
        self.layer_index = layer_index
        is_head = self.layer_index == 0
        self.is_head = is_head
        is_tail = self.layer_index == len(channel_list) - 1
        self.is_tail = is_tail
        self.in_channels = in_channels if is_head else channel_list[layer_index-1]
        self.out_channels = channel_list[layer_index]
        self.head_channels = head_channels

        if not is_head:
            self.skip_proj = nn.Sequential(
                nn.Linear(self.in_channels, self.out_channels, bias=False),
                nn.BatchNorm1d(self.out_channels, momentum=bn_momentum)
            )
            self.la = LocalAggregation(self.in_channels, self.out_channels, bn_momentum, 0.3)
            nn.init.constant_(self.skip_proj[1].weight, 0.3)

        # 7 -- 16 -- 32 -- out_channels -- out_channels
        # 3 -- 8 -- 16 -- 32 -- out_channels
        nbr_in_channels = 3 + self.in_channels if is_head else 3
        nbr_hidden_channels = 32 if is_head else 16
        nbr_out_channels = self.out_channels if is_head else 32
        self.nbr_embed = nn.Sequential(
            nn.Linear(nbr_in_channels, nbr_hidden_channels // 2, bias=False),
            nn.BatchNorm1d(nbr_hidden_channels // 2, momentum=bn_momentum),
            nn.GELU(),
            nn.Linear(nbr_hidden_channels // 2, nbr_hidden_channels, bias=False),
            nn.BatchNorm1d(nbr_hidden_channels, momentum=bn_momentum),
            nn.GELU(),
            nn.Linear(nbr_hidden_channels, nbr_out_channels, bias=False),
        )
        self.nbr_proj = nn.Identity() if is_head else nn.Linear(nbr_out_channels, self.out_channels, bias=False)
        self.nbr_bn = nn.BatchNorm1d(self.out_channels, momentum=bn_momentum)
        nn.init.constant_(self.nbr_bn.weight, 0.8 if is_head else 0.2)

        self.res_mlp = InvResMLP(
            channels=self.out_channels,
            res_blocks=res_blocks[layer_index],
            mlp_ratio=mlp_ratio,
            bn_momentum=bn_momentum,
            drop_path=drop_paths[layer_index],
        )

        mamba_config.n_layer = mamba_blocks[layer_index]
        mamba_config.d_intermediate = 0
        self.pm = PointMambaLayer(
            layer_index=layer_index,
            channels=self.out_channels,
            config=mamba_config,
            hybrid_args=hybrid_args,
            bn_momentum=bn_momentum,
        )

        self.cor_std = 1 / cross_std[layer_index]
        self.cor_head = nn.Sequential(
            nn.Linear(self.out_channels, 32, bias=False),
            nn.BatchNorm1d(32, momentum=bn_momentum),
            nn.GELU(),
            nn.Linear(32, 3, bias=False),
        )

        self.post_proj = nn.Sequential(
            nn.BatchNorm1d(self.out_channels, momentum=bn_momentum),
            nn.Linear(self.out_channels, head_channels, bias=False),
        )
        nn.init.constant_(self.post_proj[0].weight, (channel_list[0] / self.out_channels) ** 0.5)
        self.head_drop = DropPath(head_drops[layer_index])

        if not is_tail:
            self.sub_stage = Stage(
                layer_index=layer_index + 1,
                in_channels=in_channels,
                channel_list=channel_list,
                head_channels=head_channels,
                mamba_blocks=mamba_blocks,
                res_blocks=res_blocks,
                mlp_ratio=mlp_ratio,
                bn_momentum=bn_momentum,
                drop_paths=drop_paths,
                head_drops=head_drops,
                mamba_config=mamba_config,
                hybrid_args=hybrid_args,
                cross_std=cross_std
            )

    def forward(self, p, p_gs, f, gs: NaiveGaussian3D):
        assert len(f.shape) == 2
        if not self.is_head:
            p, idx = gs.gs_points.down_sampling('p', self.layer_index - 1, need_idx=True)
            p_gs = p_gs[idx]
            pre_group_idx = gs.gs_points.idx_group[self.layer_index - 1]
            f = self.skip_proj(f)[idx] + self.la(f.unsqueeze(0), pre_group_idx.unsqueeze(0)).squeeze(0)[idx]

        p_nbr, group_idx = gs.gs_points.grouping('p', self.layer_index, need_idx=True)
        p_nbr = p_nbr - p.unsqueeze(1)
        if self.is_head:
            f_nbr = f[group_idx]
            f_nbr = torch.cat([p_nbr, f_nbr], dim=-1).view(-1, 3 + self.in_channels)
        else:
            f_nbr = p_nbr.view(-1, 3)

        N, K = group_idx.shape
        nbr_embed_fn = lambda f: self.nbr_embed(f).view(N, K, -1).max(dim=1)[0]
        f_nbr = nbr_embed_fn(f_nbr)
        f_nbr = self.nbr_proj(f_nbr)
        f_nbr = self.nbr_bn(f_nbr)
        f = f_nbr if self.is_head else f_nbr + f

        pts = gs.gs_points.pts_list[self.layer_index]
        f = self.res_mlp(f, group_idx, pts.tolist())

        f_out = self.pm(p, p_gs, f, gs)

        if not self.is_tail:
            f_out_sub, c_sub = self.sub_stage(p, p_gs, f, gs)
        else:
            f_out_sub = c_sub = None

        # regularization
        if self.training:
            rel_k = torch.randint(K, (N, 1), device=f.device)
            rel_k = torch.gather(group_idx.squeeze(0), 1, rel_k).squeeze(1)
            rel_cor = (p[rel_k] - p)
            rel_cor.mul_(self.cor_std)
            rel_p = f[rel_k] - f
            rel_p = self.cor_head(rel_p)
            closs = F.mse_loss(rel_p, rel_cor)
            c_sub = c_sub + closs if c_sub is not None else closs

        f_out = self.post_proj(f_out)
        if not self.is_head:
            us_idx = gs.gs_points.idx_us[self.layer_index - 1]
            f_out = f_out[us_idx]
        f_out = self.head_drop(f_out)
        f_out = f_out_sub + f_out if f_out_sub is not None else f_out
        return f_out, c_sub


class SegHead(nn.Module):
    def __init__(self,
                 stage: Stage,
                 num_classes=13,
                 bn_momentum=0.02,
                 **kwargs
                 ):
        super().__init__()
        self.stage = stage

        self.head = nn.Sequential(
            nn.BatchNorm1d(stage.head_channels, momentum=bn_momentum),
            nn.GELU(),
            nn.Linear(stage.head_channels, num_classes),
        )

        self.apply(self.__init_weights)

    def __init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, gs: NaiveGaussian3D):
        p = gs.gs_points.p
        p_gs = gs.gs_points.p_gs
        f = gs.gs_points.f
        f, closs = self.stage(p, p_gs, f, gs)
        if self.training:
            return self.head(f), closs
        return self.head(f)
