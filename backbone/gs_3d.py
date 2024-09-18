from dataclasses import dataclass, field

import torch
from fsspec.registry import default

# from pytorch3d.ops import sample_farthest_points

from backbone.ops.gaussian_splatting_batch import project_points, compute_cov3d, ewa_project
from backbone.ops import points_centroid, points_scaler
from backbone.ops.camera import OrbitCamera


def create_sampler(sampler='random', **kwargs):
    if sampler == 'random':
        return random_sample
    # elif sampler == 'fps':
    #     return fps_sample
    else:
        raise NotImplementedError(
            f'sampler {sampler} not implemented')


# def fps_sample(xyz, n_samples, **kwargs):
#     """
#     :param xyz: [B, N, 3]
#     :return: [B, n_samples, 3], [B, n_samples]
#     """
#     random_start_point = kwargs.get('random_start', False)
#     xyz_sampled, xyz_idx = sample_farthest_points(
#         points=xyz,
#         K=n_samples,
#         random_start_point=random_start_point,
#     )
#     return xyz_sampled, xyz_idx


def random_sample(xyz, n_samples, **kwargs):
    """
    :param xyz: [B, N, 3]
    :return: [B, n_samples, 3], [B, n_samples]
    """
    B, N, _ = xyz.shape
    xyz_idx = torch.randint(0, N, (B, n_samples), device=xyz.device)
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
    cam_sampler: str = 'random'
    # generate camera method
    cam_gen_method: str = 'centroid'  # ['centroid', 'farthest']

    @classmethod
    def default(cls):
        return GaussianOptions(
            n_cameras=4,
            cam_fovy=60.0,
            cam_field_size=[512, 512],
            cam_sampler='random',
            cam_gen_method='centroid'
        )


class GaussianPoints(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

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

        self.gs_points = GaussianPoints(batch_size)

    def init_points(self):
        self.gs_points = GaussianPoints(self.batch_size)

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
                camid=2 * j,
                width=cam_width,
                height=cam_height,
                campos=(x, y, z),
                target=(cx, cy, cz),
                fovy=cam_fovy,
                device=self.device,
            ))
            inside_cameras.append(OrbitCamera(
                camid=2 * j + 1,
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
                camid=2 * j,
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
    def projects(self, xyz, cam_seed=0, scale=2.):
        """
        :param xyz: [N, 3]
        :param cam_seed: seed to generate camera id
        :param scale: xyz scale factor
        :return: [N, 3]
        """
        cam_seed = cam_seed % self.batch_size
        xyz_scaled = points_scaler(xyz, scale=scale)
        cameras = self.generate_cameras(xyz_scaled)
        n_cameras = self.opt.n_cameras
        cam_width, cam_height = self.opt.cam_field_size

        uv_all, depths_all, visible_all, camid_all = [], [], [], []
        cam_intr_all, cam_extr_all = [], []
        for j in range(n_cameras * 2):
            cam_intr = cameras[j].intrinsics
            cam_extr = cameras[j].pose
            uv, depths = project_points(
                # the batch size is 1
                xyz_scaled.unsqueeze(0),
                cam_intr.unsqueeze(0),
                cam_extr.unsqueeze(0),
                cam_width,
                cam_height,
            )
            uv = uv.squeeze(0)
            depths = depths.squeeze(0)

            visible = depths != 0
            camid = torch.zeros_like(depths, device=self.device)
            camid[...] = cameras[j].camid + cam_seed * n_cameras * 2

            uv_all.append(uv)
            depths_all.append(depths)
            visible_all.append(visible)
            camid_all.append(camid)
            cam_intr_all.append(cam_intr)
            cam_extr_all.append(cam_extr)
        uv = torch.stack(uv_all, dim=-1)
        depths = torch.stack(depths_all, dim=-1)
        visible = torch.stack(visible_all, dim=-1)
        camid = torch.stack(camid_all, dim=-1)
        cam_intr = torch.stack(cam_intr_all, dim=-1)
        cam_extr = torch.stack(cam_extr_all, dim=-1)
        self.gs_points.__update_attr__('uv', uv)
        self.gs_points.__update_attr__('depths', depths)
        self.gs_points.__update_attr__('visible', visible)
        self.gs_points.__update_attr__('camid', camid)
        self.gs_points.__update_attr__('cam_intr', cam_intr)
        self.gs_points.__update_attr__('cam_extr', cam_extr)

        # positions in gs space
        depths = points_scaler(self.gs_points.depths, scale=2.0)
        uvc = torch.cat([uv.mul(depths), camid.mul(visible)], dim=1)
        p_gs = uvc.mean(dim=-1).squeeze(-1)  # [N, 3]
        self.gs_points.__update_attr__('p_gs', p_gs)
        return p_gs

    @torch.no_grad()
    def cov3d(self, xyz_padded):
        """
        :param xyz_padded: [N, n_neighbors, 3]
        :return: [N, 3, 3]
        """
        visible = self.gs_points.visible
        visible = torch.reshape(visible, (self.batch_size, -1, 1))
        xyz_padded = torch.reshape(xyz_padded, (self.batch_size, -1, xyz_padded.shape[1], 3))
        cov3d = compute_cov3d(xyz_padded, visible)
        cov3d = torch.reshape(cov3d, (-1, 3, 3))
        return cov3d

    @torch.no_grad()
    def cov2d(self, xyz, cov3d):
        """
        :param xyz: [N, 3]
        :param cov3d: [N, 3, 3]
        :return: [N, 3, n_cameras*2]
        """
        n_cameras = self.opt.n_cameras
        cam_width, cam_height = self.opt.cam_field_size
        uv = self.gs_points.uv
        visible = self.gs_points.visible
        cam_intr = self.gs_points.cam_intr
        cam_extr = self.gs_points.cam_extr

        xyz = torch.reshape(xyz, (self.batch_size, -1, 3))
        uv = torch.reshape(uv, (self.batch_size, -1, 2))
        visible = torch.reshape(visible, (self.batch_size, -1, 1))
        cam_intr = torch.repeat_interleave(cam_intr, self.batch_size, dim=0)
        cam_extr = torch.repeat_interleave(cam_extr, self.batch_size, dim=0)

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
        cov2d = torch.stack(cov2d_all, dim=-1)
        cov2d = torch.reshape(cov2d, (-1, 3, n_cameras * 2))
        return cov2d
