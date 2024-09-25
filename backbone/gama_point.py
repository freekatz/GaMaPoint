import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from timm.models.layers import DropPath

from backbone.ops import points_scaler
from backbone.gs_3d import NaiveGaussian3D
from backbone.layers import LocalAggregation, InvResMLP, PointMambaLayer, SpatialEmbedding
from backbone.mamba_ssm.models import MambaConfig


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
                 task_type='seg',
                 **kwargs
                 ):
        super().__init__()
        assert task_type.lower() in ['seg', 'cls']
        self.task_type = task_type.lower()
        self.layer_index = layer_index
        is_head = self.layer_index == 0
        self.is_head = is_head
        is_tail = self.layer_index == len(channel_list) - 1
        self.is_tail = is_tail
        self.in_channels = in_channels if is_head else channel_list[layer_index - 1]
        self.out_channels = channel_list[layer_index]
        self.head_channels = head_channels

        if not is_head:
            self.skip_proj = nn.Sequential(
                nn.Linear(self.in_channels, self.out_channels, bias=False),
                nn.BatchNorm1d(self.out_channels, momentum=bn_momentum)
            )
            self.la = LocalAggregation(self.in_channels, self.out_channels, bn_momentum, 0.3)
            nn.init.constant_(self.skip_proj[1].weight, 0.3)

        nbr_in_channels = 3 + 3 + self.in_channels if is_head else 3 + 3
        nbr_hidden_channels = channel_list[0] if is_head else channel_list[0]//2
        nbr_out_channels = self.out_channels if is_head else channel_list[0]
        self.spe = SpatialEmbedding(
            in_channels=nbr_in_channels,
            hidden_channels=nbr_hidden_channels,
            embed_channels=nbr_out_channels,
            out_channels=self.out_channels,
            bn_momentum=bn_momentum,
        )
        self.spe_gs = SpatialEmbedding(
            in_channels=nbr_in_channels,
            hidden_channels=nbr_hidden_channels,
            embed_channels=nbr_out_channels,
            out_channels=self.out_channels,
            bn_momentum=bn_momentum,
        )
        self.alpha = nn.Parameter(torch.ones((1,), dtype=torch.float32) * 100)

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
        if self.task_type == 'seg':
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
            )

    def forward(self, p, p_gs, f, gs: NaiveGaussian3D):
        assert len(f.shape) == 2
        if not self.is_head:
            idx = gs.gs_points.idx_ds[self.layer_index - 1]
            p = p[idx]
            p_gs = p_gs[idx]
            pre_group_idx = gs.gs_points.idx_group[self.layer_index - 1]
            pre_gs_group_idx = gs.gs_points.idx_group[self.layer_index - 1]
            f = self.skip_proj(f)[idx] + self.la(f.unsqueeze(0), pre_group_idx.unsqueeze(0),
                                                 pre_gs_group_idx.unsqueeze(0)).squeeze(0)[idx]

        group_idx = gs.gs_points.idx_group[self.layer_index]
        gs_group_idx = gs.gs_points.idx_gs_group[self.layer_index]
        p_group = p[group_idx]
        p_group = p_group - p.unsqueeze(1)
        if self.is_head:
            f_group = f[group_idx]
            f_gs_group = f[gs_group_idx]
            p_gs_group = p_gs[gs_group_idx]
            f_group = torch.cat([p_group, p_gs_group, f_group + f_gs_group], dim=-1).view(-1, 3 + 3 + self.in_channels)
        else:
            p_gs_group = p_gs[gs_group_idx]
            f_group = torch.cat([p_group, p_gs_group], dim=-1).view(-1, 3 + 3)

        f_group = self.spe(f_group, group_idx)
        f = f_group if self.is_head else f_group + f

        pts = gs.gs_points.pts_list[self.layer_index]
        f = self.res_mlp(f.unsqueeze(0), group_idx.unsqueeze(0), gs_group_idx.unsqueeze(0), pts.tolist()).squeeze(0)
        f = self.pm(p, f, gs)
        if not self.is_tail:
            f_sub = self.sub_stage(p, p_gs, f, gs)
        else:
            f_sub = None

        if self.task_type == 'seg':
            f = self.post_proj(f)
            if not self.is_head:
                us_idx = gs.gs_points.idx_us[self.layer_index - 1]
                f = f[us_idx]
            f = self.head_drop(f)
        f = f_sub + f if f_sub is not None else f
        return f


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

        p = p.mul_(40)  # make diff from p and f
        f = self.stage(p, p_gs, f, gs)
        return self.head(f)


class ClsHead(nn.Module):
    def __init__(self,
                 stage: Stage,
                 num_classes=13,
                 bn_momentum=0.02,
                 **kwargs
                 ):
        super().__init__()
        self.stage = stage

        self.proj = nn.Sequential(
            nn.BatchNorm1d(stage.out_channels, momentum=bn_momentum),
            nn.Linear(stage.out_channels, stage.head_channels),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Linear(stage.head_channels, 512, bias=False),
            nn.BatchNorm1d(512, momentum=bn_momentum),
            nn.GELU(),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256, momentum=bn_momentum),
            nn.GELU(),
            nn.Dropout(.5),
            nn.Linear(256, num_classes)
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

        p = p.mul_(40)  # make diff from p and f
        f = self.stage(p, p_gs, f, gs)
        f = self.proj(f)
        f = f.max(dim=0)[0].unsqueeze(0)
        return self.head(f)

