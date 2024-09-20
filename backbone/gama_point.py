import torch
from torch import nn
from torch.nn.init import trunc_normal_

from backbone.mamba_ssm.models import MambaConfig
from backbone.gs_3d import NaiveGaussian3D
from backbone.layers import InvResMLP, PointMambaLayer, SpatialEmbedding, LocalAggregation


class Stage(nn.Module):
    def __init__(self,
                 layer_index=0,
                 in_channels=4,
                 channel_list=[64, 128, 256, 512],
                 out_channels=256,
                 mamba_blocks=[1, 1, 2, 1],
                 res_blocks=[4, 4, 8, 4],
                 mlp_ratio=2.,
                 bn_momentum=0.,
                 mamba_config=MambaConfig().default(),
                 hybrid_args={'hybrid': False, 'type': 'post', 'ratio': 0.5},
                 task_type='seg',
                 **kwargs
                 ):
        super().__init__()
        self.layer_index = layer_index
        self.in_channels = in_channels
        self.channel_list = channel_list
        self.n_layers = len(channel_list)
        self.mamba_blocks = mamba_blocks
        self.res_blocks = res_blocks
        self.mlp_ratio = mlp_ratio
        self.bn_momentum = bn_momentum
        self.mamba_config = mamba_config
        self.hybrid_args = hybrid_args
        self.out_channels = out_channels
        self.task_type = task_type
        assert self.task_type in ['seg', 'cls']

        is_head = self.layer_index == 0
        is_tail = self.layer_index == self.n_layers - 1
        self.is_head = is_head
        self.is_tail = is_tail

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

        res_blocks = self.res_blocks[layer_index]
        self.res_mlp = InvResMLP(
            channels=out_channels,
            res_blocks=res_blocks,
            mlp_ratio=self.mlp_ratio,
            bn_momentum=self.bn_momentum,
        )

        mamba_config = self.mamba_config
        mamba_config.n_layer = self.mamba_blocks[layer_index]
        mamba_config.d_intermediate = 0
        self.pm = PointMambaLayer(
            layer_index=layer_index,
            channels=out_channels,
            config=mamba_config,
            hybrid_args=self.hybrid_args,
        )

        if not self.is_tail:
            self.sub_stage = Stage(
                layer_index=layer_index + 1,
            )
        if self.task_type == 'seg':
            self.decoders = nn.Sequential(
                nn.BatchNorm1d(self.channel_list[layer_index], momentum=self.bn_momentum),
                nn.Linear(self.channel_list[layer_index], self.out_channels, bias=False),
            )
            nn.init.constant_(self.decoders[0].weight, 0.25)

    def forward(self, p, p_gs, f, gs: NaiveGaussian3D):
        if not self.is_head:
            # down sample
            p, idx = gs.gs_points.down_sampling('p', self.layer_index-1, need_idx=True)
            p_gs = p_gs[idx]
            pre_group_idx = gs.gs_points.idx_group[self.layer_index-1]
            f = self.skip_proj(f)[idx] + self.la(f.unsqueeze(0), pre_group_idx.unsqueeze(0)).squeeze(0)[idx]

        # 1. set abstraction
        p_nbr, group_idx = gs.gs_points.grouping('p', self.layer_index, need_idx=True)
        p_nbr = p_nbr - p.unsqueeze(1)
        if self.is_head:
            f_nbr = f[group_idx]
            f_nbr = torch.cat([p_nbr, f_nbr], dim=-1).view(-1, 3 + self.in_channels)
        else:
            f_nbr = p_nbr.view(-1, 3)

        f_nbr = self.spe(f_nbr, group_idx)
        f = f_nbr if self.is_head else f + f_nbr

        # 2. local aggregation
        res_mlp = self.encoders[1]
        group_idx = gs.gs_points.idx_group[self.layer_index]
        f = res_mlp(f, group_idx)

        # 3. global propagation
        if not self.is_head:
            pm = self.encoders[2]
            f_out = pm(p, p_gs, f, gs)
        else:
            f_out = f

        # 4. sub stage
        if not self.is_tail:
            f_out_sub = self.sub_stage(p, p_gs, f, gs)
        else:
            f_out_sub = None

        if self.task_type == 'seg':
            # 5. up sampling
            f_out = self.decoders(f_out)
            if not self.is_head:
                us_idx = gs.gs_points.idx_us[self.layer_index-1]
                f_out = f_out[us_idx]
            f_out = f_out + f_out_sub if f_out_sub is not None else f_out
        return f_out


class SegHead(nn.Module):
    def __init__(self,
                 stage: Stage,
                 num_classes=13,
                 bn_momentum=0.,
                 **kwargs
                 ):
        super().__init__()
        self.stage = stage

        self.head = nn.Sequential(
            nn.BatchNorm1d(self.stage.out_channels, momentum=bn_momentum),
            nn.GELU(),
            nn.Linear(self.stage.out_channels, num_classes),
        )

        self.apply(self.__init_weights)

    def __init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, gs):
        p = gs.gs_points.p
        p_gs = gs.gs_points.p_gs
        f = gs.gs_points.f
        f = self.stage(p, p_gs, f, gs)
        return self.head(f)


# class ClsHead(nn.Module):
#     def __init__(self,
#                  stage: Stage,
#                  num_classes=13,
#                  bn_momentum=0.,
#                  **kwargs
#                  ):
#         super().__init__()
#         self.stage = stage
#
#         self.head = nn.Sequential(
#             nn.BatchNorm1d(stage.out_channels, momentum=bn_momentum),
#             nn.GELU(),
#             nn.Linear(stage.out_channels, num_classes),
#         )
#
#         self.apply(self._init_weights)
#
#     def __init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, gs):
#         _, f = self.encoder.forward_cls_feat(gs)
#         return self.head(f)

