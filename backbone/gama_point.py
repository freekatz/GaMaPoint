from torch import nn
from torch.nn.init import trunc_normal_

from backbone.mamba_ssm.models import MambaConfig
from backbone.gs_3d import NaiveGaussian3D
from backbone.layers import SetAbstraction, InvResMLP, PointMambaLayer


class Encoder(nn.Module):
    def __init__(self,
                 in_channels=4,
                 channel_list=[64, 128, 256, 512],
                 mamba_blocks=[2, 2, 4, 2],
                 res_blocks=[4, 4, 8, 4],
                 mlp_ratio=2.,
                 bn_momentum=0.,
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

        res_blocks = self.res_blocks[layer_index]
        for i in range(0, res_blocks):
            encoder.append(InvResMLP(
                channels=out_channels,
                mlp_ratio=self.mlp_ratio,
                bn_momentum=self.bn_momentum,
            ))

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
        f = gs.gs_points.f
        f_out = f
        for layer_idx in range(0, self.n_layers):
            sa = self.encoders[layer_idx][0]
            if layer_idx > 0:
                # down sample
                p, idx = gs.gs_points.down_sampling('p', layer_idx, need_idx=True)
                f = f[idx]

            # 1. set abstraction
            f = sa(f, gs)

            # 2. local aggregation
            if self.res_blocks[layer_idx] > 0:
                inv_mlp = self.encoders[layer_idx][1:-1]
                group_idx = gs.gs_points.idx_group[layer_idx]
                for mlp in inv_mlp:
                    f = mlp(f, group_idx)

            # 3. global propagation
            pm = self.encoders[layer_idx][-1]
            f_out = pm(f, gs)
        return p, f_out

    def forward_seg_feat(self, gs: NaiveGaussian3D):
        p = gs.gs_points.p
        f = gs.gs_points.f
        p_list, f_list = [], []
        for layer_idx in range(0, self.n_layers):
            sa = self.encoders[layer_idx][0]
            if layer_idx > 0:
                # down sample
                p, idx = gs.gs_points.down_sampling('p', layer_idx, need_idx=True)
                f = f[idx]

            # 1. set abstraction
            f = sa(f, gs)

            # 2. local aggregation
            if self.res_blocks[layer_idx] > 0:
                inv_mlp = self.encoders[layer_idx][1:-1]
                group_idx = gs.gs_points.idx_group[layer_idx]
                for mlp in inv_mlp:
                    f = mlp(f, group_idx)

            # 3. global propagation
            pm = self.encoders[layer_idx][-1]
            f_out = pm(f, gs)

            f_list.append(f_out)
            p_list.append(p)
        return p_list, f_list

    def forward(self, gs):
        return self.forward_seg_feat(gs)


class Decoder(nn.Module):
    def __init__(self,
                 channel_list=[64, 128, 256, 512],
                 bn_momentum=0.,
                 **kwargs
                 ):
        super().__init__()
        self.channel_list = channel_list
        self.n_layers = len(channel_list)
        self.bn_momentum = bn_momentum

        self.decoders = nn.ModuleList([
            self.__make_decode_layer(layer_index=layer_index)
            for layer_index in range(1, self.n_layers)
        ])
        self.out_channels = channel_list[-1]

    def __make_decode_layer(self, layer_index):
        in_channels = self.channel_list[layer_index]
        out_channels = self.channel_list[layer_index-1]
        proj = nn.Sequential(
            nn.BatchNorm1d(in_channels, momentum=self.bn_momentum),
            nn.Linear(in_channels, out_channels, bias=False),
        )
        nn.init.constant_(proj[0].weight, 0.25)
        return proj

    def forward(self, p_list, f_list, gs: NaiveGaussian3D):
        for layer_idx in range(0, self.n_layers):
            p, f = p_list[-layer_idx-1], f_list[-layer_idx-1]
            if layer_idx < self.n_layers - 1:
                p, us_idx = gs.gs_points.up_sampling('p', layer_idx, need_idx=True)
                f = f[us_idx]
            if layer_idx > 0:
                f = self.decoder[layer_idx](f) + f_list[-layer_idx-2]
                f_list[-layer_idx-1] = f
        for i in range(-1, -len(self.decoders) - 1, -1):
            us_idx = gs.gs_points.idx_us[-len(self.decoder) - i - 1]
            f_list[i-1] = f_list[i-1] + self.decoders[i](f_list[i][us_idx])
        return f_list[-len(self.decoder) - 1]


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

