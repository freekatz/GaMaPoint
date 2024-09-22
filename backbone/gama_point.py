import torch
from timm.layers import DropPath

import __init__

from torch import nn
from torch.nn.init import trunc_normal_

from backbone.mamba_ssm.models import MambaConfig
from backbone.gs_3d import NaiveGaussian3D
from backbone.layers import SetAbstraction, InvResMLP, PointMambaLayer


class Encoder(nn.Module):
    def __init__(self,
                 in_channels=4,
                 channel_list=[64, 128, 256, 512],
                 mamba_blocks=[1, 1, 2, 1],
                 res_blocks=[4, 4, 8, 4],
                 mlp_ratio=2.,
                 bn_momentum=0.,
                 drop_paths=None,
                 mamba_config=MambaConfig().default(),
                 hybrid_args={'hybrid': False, 'type': 'post', 'ratio': 0.5},
                 **kwargs
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.channel_list = channel_list
        self.out_channels = channel_list[-1]
        self.n_layers = len(channel_list)
        self.mamba_blocks = mamba_blocks
        self.res_blocks = res_blocks
        self.mlp_ratio = mlp_ratio
        self.bn_momentum = bn_momentum
        assert drop_paths is not None
        self.drop_paths = drop_paths
        self.mamba_config = mamba_config
        self.hybrid_args = hybrid_args

        self.encoders = nn.ModuleList([
            self.__make_encode_layer(layer_index=layer_index)
            for layer_index in range(self.n_layers)
        ])

    def __make_encode_layer(self, layer_index) -> nn.ModuleList:
        in_channels = self.in_channels
        if layer_index > 0:
            in_channels = self.channel_list[layer_index - 1]
        out_channels = self.channel_list[layer_index]
        encoder = []

        sa = SetAbstraction(
            layer_index=layer_index,
            in_channels=in_channels,
            out_channels=out_channels,
            bn_momentum=self.bn_momentum,
        )
        encoder.append(sa)

        res_mlp = InvResMLP(
            channels=out_channels,
            res_blocks=self.res_blocks[layer_index],
            mlp_ratio=self.mlp_ratio,
            bn_momentum=self.bn_momentum,
            drop_path=self.drop_paths[layer_index],
        )
        encoder.append(res_mlp)

        mamba_config = self.mamba_config
        mamba_config.n_layer = self.mamba_blocks[layer_index]
        mamba_config.d_intermediate = 0
        pm = PointMambaLayer(
            layer_index=layer_index,
            channels=out_channels,
            config=mamba_config,
            hybrid_args=self.hybrid_args,
        )
        encoder.append(pm)
        return nn.ModuleList(encoder)

    def forward_cls_feat(self, gs: NaiveGaussian3D):
        p = gs.gs_points.p
        p_gs = gs.gs_points.p_gs
        f = gs.gs_points.f
        f_out = f
        for layer_idx in range(0, self.n_layers):
            sa = self.encoders[layer_idx][0]
            if layer_idx > 0:
                # down sample
                p, idx = gs.gs_points.down_sampling('p', layer_idx-1, need_idx=True)
                p_gs = p_gs[idx]

            # 1. set abstraction
            f = sa(p, f, gs)

            # 2. local aggregation
            res_mlp = self.encoders[layer_idx][1]
            group_idx = gs.gs_points.idx_group[layer_idx]
            f = res_mlp(f, group_idx)

            # 3. global propagation
            if layer_idx > 0:
                # pm = self.encoders[layer_idx][2]
                # f_out = pm(p, p_gs, f, gs)
                f_out = f
            else:
                f_out = f
        return p, f_out

    def forward_seg_feat(self, gs: NaiveGaussian3D):
        p = gs.gs_points.p
        p_gs = gs.gs_points.p_gs
        f = gs.gs_points.f
        p_list, f_list = [p], [f]
        for layer_idx in range(0, self.n_layers):
            sa = self.encoders[layer_idx][0]
            if layer_idx > 0:
                # down sample
                p, idx = gs.gs_points.down_sampling('p', layer_idx-1, need_idx=True)
                p_gs = p_gs[idx]

            # 1. set abstraction
            f = sa(p, f, gs)

            # 2. local aggregation
            res_mlp = self.encoders[layer_idx][1]
            group_idx = gs.gs_points.idx_group[layer_idx]
            pts = gs.gs_points.pts_list[layer_idx]
            f = res_mlp(f, group_idx, pts)

            # 3. global propagation
            if layer_idx > 0:
                pm = self.encoders[layer_idx][2]
                f_out = pm(p, p_gs, f, gs)
            else:
                f_out = f

            f_list.append(f_out)
            p_list.append(p)
        return p_list, f_list

    def forward(self, gs):
        return self.forward_seg_feat(gs)


class Decoder(nn.Module):
    def __init__(self,
                 channel_list=[64, 128, 256, 512],
                 out_channels=256,
                 bn_momentum=0.,
                 head_drops=None,
                 **kwargs
                 ):
        super().__init__()
        self.channel_list = channel_list[::-1]
        self.out_channels = out_channels
        self.n_layers = len(channel_list)
        self.bn_momentum = bn_momentum
        assert head_drops is not None
        self.head_drops = head_drops

        self.decoders = nn.ModuleList([
            self.__make_decode_layer(layer_index=layer_index)
            for layer_index in range(0, self.n_layers)
        ])

    def __make_decode_layer(self, layer_index):
        in_channels = self.channel_list[layer_index]
        out_channels = self.out_channels
        decoder = []

        proj = nn.Sequential(
            nn.BatchNorm1d(in_channels, momentum=self.bn_momentum),
            nn.Linear(in_channels, out_channels, bias=False),
        )
        decoder.append(proj)
        nn.init.constant_(proj[0].weight, 0.25)

        drop = DropPath(self.head_drops[layer_index])
        decoder.append(drop)
        return nn.ModuleList(decoder)

    def forward(self, p_list, f_list, gs: NaiveGaussian3D):
        p_list = p_list[::-1]
        f_list = f_list[::-1]
        idx_us = gs.gs_points.idx_us[::-1]
        for i in range(len(self.decoders)):
            proj = self.decoders[i][0]
            f_decode = proj(f_list[i])
            if i < len(self.decoders) - 1:
                us_idx = idx_us[i]
                f_decode = f_decode[us_idx]
            drop = self.decoders[i][1]
            f_decode = drop(f_decode)
            if i == 0:
                f_list[i] = f_decode
            else:
                f_list[i] = f_decode + f_list[i-1]
        return f_list[len(self.decoders)-1]


class SegHead(nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 num_classes=13,
                 bn_momentum=0.,
                 **kwargs
                 ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.head = nn.Sequential(
            nn.BatchNorm1d(decoder.out_channels, momentum=bn_momentum),
            nn.GELU(),
            nn.Linear(decoder.out_channels, num_classes),
        )

        self.apply(self.__init_weights)

    def __init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, gs):
        p_list, f_list = self.encoder.forward_seg_feat(gs)
        f = self.decoder(p_list, f_list, gs)
        return self.head(f)


class ClsHead(nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 num_classes=13,
                 bn_momentum=0.,
                 **kwargs
                 ):
        super().__init__()
        self.encoder = encoder

        self.head = nn.Sequential(
            nn.BatchNorm1d(encoder.out_channels, momentum=bn_momentum),
            nn.GELU(),
            nn.Linear(encoder.out_channels, num_classes),
        )

        self.apply(self._init_weights)

    def __init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, gs):
        _, f = self.encoder.forward_cls_feat(gs)
        return self.head(f)

