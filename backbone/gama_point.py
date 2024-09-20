from torch import nn
from torch.nn.init import trunc_normal_

from backbone.mamba_ssm.models import MambaConfig
from backbone.gs_3d import NaiveGaussian3D
from backbone.layers import SetAbstraction, InvResMLP, PointMambaLayer


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
        self.is_head = layer_index == 0
        self.is_tail = layer_index == self.n_layers - 1

        self.encoders = self.__make_encode_layer(layer_index=layer_index)
        self.decoders = self.__make_decode_layer(layer_index=layer_index)
        if not self.is_tail:
            self.sub_stage = Stage(
                layer_index=layer_index + 1,
            )

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

        res_blocks = self.res_blocks[layer_index]
        res_mlp = InvResMLP(
            channels=out_channels,
            res_blocks=res_blocks,
            mlp_ratio=self.mlp_ratio,
            bn_momentum=self.bn_momentum,
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

    def __make_decode_layer(self, layer_index):
        in_channels = self.channel_list[-layer_index-1]
        out_channels = self.out_channels
        proj = nn.Sequential(
            nn.BatchNorm1d(in_channels, momentum=self.bn_momentum),
            nn.Linear(in_channels, out_channels, bias=False),
        )
        nn.init.constant_(proj[0].weight, 0.25)
        return proj

    def forward(self, p, p_gs, f, gs: NaiveGaussian3D):
        sa = self.encoders[0]
        if not self.is_head:
            # down sample
            p, idx = gs.gs_points.down_sampling('p', self.layer_index-1, need_idx=True)
            p_gs = p_gs[idx]

        # 1. set abstraction
        f = sa(p, f, gs)

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

        # 5. up sampling
        f_out = self.decoders(f_out)
        if not self.is_head:
            us_idx = gs.gs_points.idx_us[-self.layer_idx]
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

