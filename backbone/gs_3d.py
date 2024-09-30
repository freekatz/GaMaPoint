from dataclasses import dataclass, field

from einops import repeat
from pytorch3d.ops import sample_farthest_points
import torch
from torch_kdtree import build_kd_tree

from backbone.ops.gaussian_splatting_batch import project_points, compute_cov3d, ewa_project
from backbone.ops import points_centroid, points_scaler
from backbone.ops.camera import OrbitCamera
from utils.cutils import grid_subsampling, KDTree


def create_sampler(sampler='random', **kwargs):
    if sampler == 'random':
        return random_sample
    elif sampler == 'fps':
        return fps_sample
    else:
        raise NotImplementedError(
            f'sampler {sampler} not implemented')


def fps_sample(xyz, n_samples, **kwargs):
    """
    :param xyz: [B, N, 3]
    :return: [B, n_samples, 3], [B, n_samples]
    """
    random_start_point = kwargs.get('random_start', False)
    xyz_sampled, xyz_idx = sample_farthest_points(
        points=xyz,
        K=n_samples,
        random_start_point=random_start_point,
    )
    return xyz_sampled, xyz_idx


def random_sample(xyz, n_samples, **kwargs):
    """
    :param xyz: [B, N, 3]
    :return: [B, n_samples, 3], [B, n_samples]
    """
    B, N, _ = xyz.shape
    xyz_idx = torch.randint(0, N, (B, n_samples), device=xyz.device)
    xyz_sampled = torch.gather(xyz, 1, xyz_idx.unsqueeze(-1).expand((-1, -1, 3)))
    return xyz_sampled, xyz_idx


def visible_sample(xyz, visible, n_samples, **kwargs):
    """
    :param xyz: [B, N, 3]
    :param visible: [B, N, 1, n_cameras*2]
    :return: [B, n_samples, 3], [B, n_samples]
    """
    B, N, _ = xyz.shape
    n_cameras = visible.shape[-1]
    visible = visible.sum(dim=-1, keepdim=False).squeeze(-1)
    visible = visible / n_cameras
    xyz_idx = torch.multinomial(visible.float().softmax(dim=1), n_samples, replacement=False)
    xyz_sampled = torch.gather(xyz, 1, xyz_idx.unsqueeze(-1).expand((-1, -1, 3)))
    return xyz_sampled, xyz_idx


@dataclass
class GaussianOptions(dict):
    # camera numbers outside in points
    n_cameras: int = 4
    # camera field of view in degree along y-axis.
    cam_fovy: float = 60.0
    # camera field size, [width, height]
    cam_field_size: list = field(default_factory=list)
    # camera sampler
    cam_sampler: str = 'fps'
    # generate camera method
    cam_gen_method: str = 'centroid'  # ['centroid', 'farthest']

    @classmethod
    def default(cls):
        return GaussianOptions(
            n_cameras=4,
            cam_fovy=60.0,
            cam_field_size=[512, 512],
            cam_sampler='fps',
            cam_gen_method='centroid'
        )

    def __str__(self):
        return f'''GaussianOptions(
            n_cameras={self.n_cameras},
            cam_fovy={self.cam_fovy},
            cam_field_size={self.cam_field_size},
            cam_sampler={self.cam_sampler},
            cam_gen_method={self.cam_gen_method})'''


class GaussianPoints(object):
    def __set_attr__(self, key, value):
        assert not self.__is_attr_exists__(key)
        self.__dict__[key] = value

    def __update_attr__(self, key, value):
        self.__dict__[key] = value

    def __get_attr__(self, key):
        if not self.__is_attr_exists__(key):
            return None
        return self.__dict__[key]

    def __del_attr__(self, key):
        if not self.__is_attr_exists__(key):
            return
        self.__dict__.pop(key)

    def __is_attr_exists__(self, key):
        return key in self.__dict__.keys()

    def keys(self):
        return self.__dict__.keys()

    def to_cuda(self, non_blocking=True):
        keys = self.keys()
        for key in keys:
            item = self.__get_attr__(key)
            if isinstance(self.__get_attr__(key), torch.Tensor):
                item = item.cuda(non_blocking=non_blocking)
            if isinstance(item, list):
                for i in range(len(item)):
                    if isinstance(item[i], torch.Tensor):
                        item[i] = item[i].cuda(non_blocking=non_blocking)
            self.__update_attr__(key, item)

    def apply_index(self, index):
        assert len(index.shape) == 1
        N = index.shape[0]
        keys = self.keys()
        for key in keys:
            item = self.__get_attr__(key)
            if isinstance(self.__get_attr__(key), torch.Tensor):
                if item.shape[0] >= N:
                    item = item[index]
            self.__update_attr__(key, item)

    def down_sampling(self, key, layer_idx, need_idx=False):
        item = self.__get_attr__(key)
        ds_idx = self.idx_ds[layer_idx]
        if need_idx:
            return item[ds_idx], ds_idx
        return item[ds_idx]

    def up_sampling(self, key, layer_idx, need_idx=False):
        item = self.__get_attr__(key)
        us_idx = self.idx_us[-layer_idx - 1]
        if need_idx:
            return item[us_idx], us_idx
        return item[us_idx]

    def grouping(self, key, layer_idx, need_idx=False):
        item = self.__get_attr__(key)
        group_idx = self.idx_group[layer_idx]
        if need_idx:
            return item[group_idx], group_idx
        return item[group_idx]

    def gs_grouping(self, key, layer_idx, need_idx=False):
        item = self.__get_attr__(key)
        gs_group_idx = self.idx_gs_group[layer_idx]
        if need_idx:
            return item[gs_group_idx], gs_group_idx
        return item[gs_group_idx]

    @property
    def layer_idx(self):
        return self.__get_attr__('layer_idx')

    @property
    def idx_ds(self):
        return self.__get_attr__('idx_ds')

    @property
    def idx_us(self):
        return self.__get_attr__('idx_us')

    @property
    def idx_group(self):
        return self.__get_attr__('idx_group')

    @property
    def idx_gs_group(self):
        return self.__get_attr__('idx_gs_group')

    @property
    def pts_list(self):
        return self.__get_attr__('pts_list')

    @property
    def p(self):
        return self.__get_attr__('p')

    @property
    def f(self):
        return self.__get_attr__('f')

    @property
    def y(self):
        return self.__get_attr__('y')

    @property
    def p_gs(self):
        return self.__get_attr__('p_gs')

    @property
    def cameras(self):
        cameras = self.__get_attr__('cameras')
        return cameras

    @property
    def uv(self):
        if self.layer_idx is not None:
            return self.down_sampling('uv', self.layer_idx, need_idx=False)
        return self.__get_attr__('uv')

    @property
    def depths(self):
        if self.layer_idx is not None:
            return self.down_sampling('depths', self.layer_idx, need_idx=False)
        return self.__get_attr__('depths')

    @property
    def visible(self):
        if self.layer_idx is not None:
            return self.down_sampling('visible', self.layer_idx, need_idx=False)
        return self.__get_attr__('visible')

    @property
    def camid(self):
        if self.layer_idx is not None:
            return self.down_sampling('camid', self.layer_idx, need_idx=False)
        return self.__get_attr__('camid')

    @property
    def cam_intr(self):
        return self.__get_attr__('cam_intr')

    @property
    def cam_extr(self):
        return self.__get_attr__('cam_extr')

    @property
    def cov3d(self):
        return self.__get_attr__('cov3d')

    @property
    def cov2d(self):
        return self.__get_attr__('cov2d')


class NaiveGaussian3D:
    """
    A pipeline about 3D naive gaussian to process the point cloud
    """

    def __init__(self,
                 opt: GaussianOptions = None,
                 batch_size: int = 8,
                 device: str = 'cuda',
                 **kwargs):
        if opt is None:
            opt = GaussianOptions.default()
        self.opt = opt
        self.batch_size = batch_size
        self.device = device

        self.gs_points = GaussianPoints()

    def init_points(self):
        self.gs_points = GaussianPoints()

    def generate_cameras(self, xyz):
        """
        :param xyz: [N, 3]
        :return: [n_cameras*2]
        """
        gen_method = self.opt.cam_gen_method
        if gen_method == 'centroid':
            return self.generate_cameras_by_centroid(xyz)
        elif gen_method == 'farthest':
            return self.generate_cameras_by_farthest(xyz)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def generate_cameras_by_centroid(self, xyz):
        """
        :param xyz: [N, 3]
        :return: [n_cameras*2]
        """
        n_cameras = self.opt.n_cameras
        cam_fovy = self.opt.cam_fovy
        cam_width, cam_height = self.opt.cam_field_size
        cam_sampler = create_sampler(self.opt.cam_sampler)

        centroid = points_centroid(xyz.unsqueeze(0)).squeeze(0)
        xyz_sampled, _ = cam_sampler(xyz=xyz.unsqueeze(0), n_samples=n_cameras)
        xyz_sampled = xyz_sampled.squeeze(0)
        outside_cameras = inside_cameras = []
        for j in range(n_cameras):
            cx, cy, cz = centroid
            x, y, z = xyz_sampled[j]
            outside_cameras.append(OrbitCamera(
                camid=2 * j + 1,
                width=cam_width,
                height=cam_height,
                campos=(x, y, z),
                target=(cx, cy, cz),
                fovy=cam_fovy,
                device=self.device,
            ))
            inside_cameras.append(OrbitCamera(
                camid=2 * j + 2,
                width=cam_width,
                height=cam_height,
                campos=(cx, cy, cz),
                target=(x, y, z),
                fovy=cam_fovy,
                device=self.device,
            ))
        cameras_all = outside_cameras + inside_cameras
        self.gs_points.__update_attr__('cameras', cameras_all)
        return cameras_all

    @torch.no_grad()
    def generate_cameras_by_farthest(self, xyz):
        """
        :param xyz: [N, 3]
        :return: [n_cameras*2]
        """
        n_cameras = self.opt.n_cameras
        cam_fovy = self.opt.cam_fovy
        cam_width, cam_height = self.opt.cam_field_size
        assert self.opt.cam_sampler == 'fps'
        cam_sampler = create_sampler(self.opt.cam_sampler)

        xyz_sampled, _ = cam_sampler(xyz=xyz.unsqueeze(0), n_samples=n_cameras * 2 + 1)
        xyz_sampled = xyz_sampled.squeeze(0)
        cameras_all = []
        for j in range(1, n_cameras * 2 + 1):
            cx, cy, cz = xyz_sampled[j - 1]
            x, y, z = xyz_sampled[j]
            cameras_all.append(OrbitCamera(
                camid=j,
                width=cam_width,
                height=cam_height,
                campos=(x, y, z),
                target=(cx, cy, cz),
                fovy=cam_fovy,
                device=self.device,
            ))
        self.gs_points.__update_attr__('cameras', cameras_all)
        return cameras_all

    @torch.no_grad()
    def projects(self, xyz, cam_seed=0, scale=1., cam_batch=1):
        """
        :param xyz: [N, 3]
        :param cam_seed: seed to generate camera id
        :param scale: xyz scale factor
        :param cam_batch: batch size of cameras to project
        :return: [N, 3]
        """
        assert len(xyz.shape) == 2
        cam_seed = cam_seed % self.batch_size
        n_cameras = self.opt.n_cameras
        assert n_cameras * 2 % cam_batch == 0
        cam_width, cam_height = self.opt.cam_field_size
        xyz_scaled = points_scaler(xyz.unsqueeze(0), scale=scale).squeeze(0)
        cameras = self.generate_cameras(xyz_scaled)

        uv_all, depths_all, visible_all = [], [], []
        cam_intr_all, cam_extr_all = [], []
        camid = torch.zeros((n_cameras*2, xyz_scaled.shape[0], 1), device=self.device)
        for j in range(n_cameras * 2 // cam_batch):
            cam_intr_batch = []
            cam_extr_batch = []
            for i in range(cam_batch):
                cam_intr = cameras[j*cam_batch + i].intrinsics
                cam_extr = cameras[j*cam_batch + i].pose
                camid[j*cam_batch + i, ...] = cameras[j*cam_batch + i].camid + cam_seed * n_cameras * 2
                cam_intr_batch.append(cam_intr)
                cam_extr_batch.append(cam_extr)
            cam_intr = torch.stack(cam_intr_batch, dim=0)
            cam_extr = torch.stack(cam_extr_batch, dim=0)
            uv, depths = project_points(
                # the batch size is cam_batch
                xyz_scaled,
                cam_intr,
                cam_extr,
                cam_width,
                cam_height,
            )
            visible = (depths != 0).int()

            uv_all.append(uv)
            depths_all.append(depths)
            visible_all.append(visible)
            cam_intr_all.append(cam_intr)
            cam_extr_all.append(cam_extr)

        uv = torch.cat(uv_all, dim=0).permute(1, 2, 0)
        depths = torch.cat(depths_all, dim=0).permute(1, 2, 0)
        visible = torch.cat(visible_all, dim=0).permute(1, 2, 0)
        camid = camid.permute(1, 2, 0)
        cam_intr = torch.cat(cam_intr_all, dim=0).permute(1, 0)
        cam_extr = torch.cat(cam_extr_all, dim=0).permute(1, 2, 0)

        self.gs_points.__update_attr__('uv', uv)
        self.gs_points.__update_attr__('depths', depths)
        self.gs_points.__update_attr__('visible', visible)
        self.gs_points.__update_attr__('camid', camid)
        self.gs_points.__update_attr__('cam_intr', cam_intr)
        self.gs_points.__update_attr__('cam_extr', cam_extr)

        # positions in gs space
        depths = points_scaler(depths.unsqueeze(0), scale=1.).squeeze(0)
        i = torch.arange(1, n_cameras*2+1)
        i = repeat(i, 'c -> n d c', n=camid.shape[0], d=1)
        uvc = torch.cat([uv.mul(depths), camid * visible * i], dim=1).squeeze(-1)  # [N, 3]
        p_gs = uvc.mean(dim=-1).squeeze(-1)  # [N, 3]
        p_gs = points_scaler(p_gs.unsqueeze(0), scale=1.0).squeeze(0)
        self.gs_points.__update_attr__('p_gs', p_gs)
        return p_gs

    @torch.no_grad()
    def cov3d(self, xyz_padded):
        # todo support manual idx
        """
        :param xyz_padded: [N, n_neighbors, 3]
        :return: [N, 3, 3]
        """
        visible = self.gs_points.visible
        cov3d = compute_cov3d(xyz_padded.unsqueeze(0), visible.unsqueeze(0)).squeeze(0)
        return cov3d

    @torch.no_grad()
    def cov2d(self, xyz, cov3d):
        # todo support manual idx
        """
        :param xyz: [N, 3]
        :param cov3d: [N, 3, 3]
        :return: [N, 3, n_cameras*2]
        """
        n_cameras = self.opt.n_cameras
        cam_width, cam_height = self.opt.cam_field_size
        uv = self.gs_points.uv.unsqueeze(0)
        visible = self.gs_points.visible.unsqueeze(0)
        cam_intr = self.gs_points.cam_intr.unsqueeze(0)
        cam_extr = self.gs_points.cam_extr.unsqueeze(0)
        xyz = xyz.unsqueeze(0)

        cov2d_all = []
        for j in range(n_cameras * 2):
            cov2d, _, _ = ewa_project(
                xyz=xyz,
                cov3d=cov3d,
                intr=cam_intr[:, j].squeeze(-1),
                extr=cam_extr[:, :, j].squeeze(-1),
                uv=uv[:, :, j].squeeze(-1),
                W=cam_width,
                H=cam_height,
                visible=visible[:, :, j].squeeze(-1),
            )
            cov2d_all.append(cov2d)
        cov2d = torch.stack(cov2d_all, dim=-1).squeeze(0)
        return cov2d


def make_gs_points(gs_points, ks, ks_gs, grid_size=None, strides=None, up_sample=True, visible_sample_stride=0.) -> GaussianPoints:
    assert (grid_size is not None and strides is not None) is False
    assert (grid_size is None and strides is None) is False
    n_layers = len(ks)
    p = gs_points.p
    p_gs = gs_points.p_gs


    # gs_points.apply_index(idx)

    idx_ds = []
    idx_us = []
    idx_group = []
    idx_gs_group = []

    for i in range(n_layers):
        # down sample
        if i > 0:
            if grid_size is not None:
                gsize = grid_size[i-1]
                if p.is_cuda:
                    ds_idx = grid_subsampling(p.detach().cpu(), gsize)
                else:
                    ds_idx = grid_subsampling(p, gsize)
            else:
                if visible_sample_stride > 0 and i == 1:
                    _, ds_idx = visible_sample(p.unsqueeze(0), gs_points.visible.unsqueeze(0), int(p.shape[0] // visible_sample_stride))
                    ds_idx = ds_idx.squeeze(0)
                else:
                    stride = strides[i-1]
                    _, ds_idx = fps_sample(p.unsqueeze(0), p.shape[0]//stride)
                    ds_idx = ds_idx.squeeze(0)
            p = p[ds_idx]
            p_gs = p_gs[ds_idx]
            idx_ds.append(ds_idx)

        # group
        k = ks[i]
        k_gs = ks_gs[i]
        kdt = KDTree(p)
        kdt_gs = KDTree(p_gs)
        idx_group.append(kdt.knn(p, k, False)[0].long())
        idx_gs_group.append(kdt_gs.knn(p_gs, k_gs, False)[0].long())

        # up sample
        if i > 0 and up_sample:
            us_idx = kdt.knn(gs_points.p, 1, False)[0].squeeze(-1)
            idx_us.append(us_idx)

    gs_points.__update_attr__('idx_ds', idx_ds)
    gs_points.__update_attr__('idx_us', idx_us)
    gs_points.__update_attr__('idx_group', idx_group)
    gs_points.__update_attr__('idx_gs_group', idx_gs_group)
    return gs_points


def merge_gs_list(gs_list, up_sample=True) -> NaiveGaussian3D:
    assert len(gs_list) > 0
    new_gs = NaiveGaussian3D(gs_list[0].opt, batch_size=gs_list[0].batch_size)

    p_all = []
    p_gs_all = []
    f_all = []
    y_all = []
    idx_ds_all = []
    idx_us_all = []
    idx_group_all = []
    idx_gs_group_all = []
    pts_all = []
    n_layers = len(gs_list[0].gs_points.idx_group)
    pts_per_layer = [0] * n_layers
    for i in range(len(gs_list)):
        gs = gs_list[i]
        p_all.append(gs.gs_points.p)
        p_gs_all.append(gs.gs_points.p_gs)
        f_all.append(gs.gs_points.f)
        y_all.append(gs.gs_points.y)

        idx_ds = gs.gs_points.idx_ds
        idx_us = gs.gs_points.idx_us
        idx_group = gs.gs_points.idx_group
        idx_gs_group = gs.gs_points.idx_gs_group
        pts = []
        for layer_idx in range(n_layers):
            if layer_idx < len(idx_ds):
                idx_ds[layer_idx].add_(pts_per_layer[layer_idx])
                if up_sample:
                    idx_us[layer_idx].add_(pts_per_layer[layer_idx + 1])
            idx_group[layer_idx].add_(pts_per_layer[layer_idx])
            idx_gs_group[layer_idx].add_(pts_per_layer[layer_idx])
            pts.append(idx_group[layer_idx].shape[0])
        idx_ds_all.append(idx_ds)
        idx_us_all.append(idx_us)
        idx_group_all.append(idx_group)
        idx_gs_group_all.append(idx_gs_group)
        pts_all.append(pts)
        pts_per_layer = [pt + idx.shape[0] for (pt, idx) in zip(pts_per_layer, idx_group)]

    p = torch.cat(p_all, dim=0)
    p_gs = torch.cat(p_gs_all, dim=0)
    f = torch.cat(f_all, dim=0)
    y = torch.cat(y_all, dim=0)
    idx_ds = [torch.cat(idx, dim=0) for idx in zip(*idx_ds_all)]
    idx_us = [torch.cat(idx, dim=0) for idx in zip(*idx_us_all)]
    idx_group = [torch.cat(idx, dim=0) for idx in zip(*idx_group_all)]
    idx_gs_group = [torch.cat(idx, dim=0) for idx in zip(*idx_gs_group_all)]
    new_gs.gs_points.__update_attr__('p', p)
    new_gs.gs_points.__update_attr__('p_gs', p_gs)
    new_gs.gs_points.__update_attr__('f', f)
    new_gs.gs_points.__update_attr__('y', y)
    new_gs.gs_points.__update_attr__('idx_ds', idx_ds)  # layer_idx: [1, 2, 3]
    new_gs.gs_points.__update_attr__('idx_us', idx_us)  # layer_idx: [2, 1, 0]
    new_gs.gs_points.__update_attr__('idx_group', idx_group)  # layer_idx: [0, 1, 2, 3]
    new_gs.gs_points.__update_attr__('idx_gs_group', idx_gs_group)  # layer_idx: [0, 1, 2, 3]
    pts_list = torch.tensor(pts_all, dtype=torch.int64)
    pts_list = pts_list.view(-1, n_layers).transpose(0, 1).contiguous()  # batch_size * layer_idx: [0, 1, 2, 3]
    new_gs.gs_points.__update_attr__('pts_list', pts_list)
    return new_gs

