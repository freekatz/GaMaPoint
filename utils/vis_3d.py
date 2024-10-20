import os

import cv2
import numpy as np
import torch


def write_obj(points, colors, out_filename):
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[i]
        fout.write('v %f %f %f %f %f %f\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    fout.close()


def read_obj(filename):
    values = np.loadtxt(filename, usecols=(1, 2, 3, 4, 5, 6))
    return values[:, :3], values[:, 3:6]


# show multiple point clouds at once in splitted windows.
def vis_multi_points(points, colors=None, labels=None,
                     opacity=1.0, point_size=10.0, title='title',
                     color_map='Paired', save_fig=False, save_name='example',
                     plot_shape=None, **kwargs):
    """Visualize a point cloud

    Args:
        points (list): a list of 2D numpy array.
        colors (list, optional): [description]. Defaults to None.

    Example:
        vis_multi_points([points, pts], labels=[self.sub_clouds_points_labels[cloud_ind], labels])
    """
    import pyvista as pv
    import numpy as np
    from pyvista import themes
    from matplotlib import cm

    my_theme = themes.Theme()
    my_theme.color = 'white'
    my_theme.lighting = True
    my_theme.show_edges = True
    my_theme.edge_color = 'white'
    my_theme.background = 'white'
    my_theme.allow_empty_mesh = True
    my_theme.title = title
    pv.set_plot_theme(my_theme)

    n_clouds = len(points)
    if plot_shape is None:
        plot_shape = (1, n_clouds)
    plotter = pv.Plotter(shape=plot_shape, border=False, **kwargs)

    shape_x, shape_y = plot_shape

    if colors is None:
        colors = [None] * n_clouds
    if labels is None:
        labels = [None] * n_clouds

    idx = -1
    for i in range(shape_x):
        for j in range(shape_y):
            idx += 1
            if idx >= n_clouds:
                break
            plotter.subplot(i, j)
            if len(points[idx].shape) == 3: points[idx] = points[idx][0]
            if colors[idx] is not None and len(colors[idx].shape) == 3: colors[idx] = colors[idx][0]
            if colors[idx] is None and labels[idx] is not None:
                color_maps = cm.get_cmap(color_map, labels[idx].max() + 1)
                colors[idx] = color_maps(labels[idx])[:, :3]
                if colors[idx].min() < 0:
                    colors[idx] = np.array(
                        (colors[idx] - colors[idx].min) / (colors[idx].max() - colors[idx].min()) * 255).astype(
                        np.int8)
            plotter.add_points(points[idx], opacity=opacity, point_size=point_size, render_points_as_spheres=True,
                               scalars=colors[idx], rgb=True, style='points')
    # plotter.link_views() # pyvista might have bug for linked_views. Comment this line out if you cannot see the visualzation result.
    if save_fig:
        plotter.show(screenshot=f'{save_name}.png')
        plotter.close()
    else:
        plotter.show()


def calc_cmap(labels):
    max_pixel = np.max(labels)
    min_pixel = np.min(labels)
    delta = max_pixel - min_pixel
    cmap = ((labels - min_pixel) / delta * 255)
    cmap = cmap * (-1)
    cmap = cmap + 255
    cmap = cmap.astype(np.uint8)
    return cmap


red = torch.tensor([1., 0., 0.])
blue = torch.tensor([0., 0., 1.])
black = torch.tensor([0, 0, 0])
white = torch.tensor([1., 1., 1.])
gray = torch.tensor([128 / 255, 128 / 255, 128 / 255])
yellow = torch.tensor([1., 1., 0])
green = torch.tensor([0., 128 / 255, 0.])


def vis_knn(p, p_idx, group_idx, **kwargs):
    group_idx = group_idx.long()
    colors = torch.zeros_like(p, dtype=torch.float, device=p.device)
    colors[:] = white
    g_idx = group_idx[p_idx]
    colors[g_idx] = blue
    colors[p_idx] = red
    vis_multi_points(
        [p.detach().cpu().numpy()],
        [colors.detach().cpu().numpy()],
        plot_shape=(1, 1), **kwargs
    )


def vis_knn2(p, p_idx, group_idx_1, group_idx_2, **kwargs):
    group_idx_1 = group_idx_1.long()
    group_idx_2 = group_idx_2.long()
    colors_1 = torch.zeros_like(p, dtype=torch.float, device=p.device)
    colors_2 = torch.zeros_like(p, dtype=torch.float, device=p.device)
    colors_1[:] = white
    colors_2[:] = white
    g_idx_1 = group_idx_1[p_idx].unsqueeze(-1).expand(-1, 3)
    g_idx_2 = group_idx_2[p_idx].unsqueeze(-1).expand(-1, 3)
    g_idx_3 = g_idx_1[torch.isin(g_idx_1, g_idx_2)]
    colors_1[g_idx_1] = blue
    colors_2[g_idx_2] = yellow
    colors_1[g_idx_3] = green  # same
    colors_2[g_idx_3] = green  # same
    colors_1[p_idx] = red
    colors_2[p_idx] = red
    vis_multi_points(
        [p.detach().cpu().numpy(), p.detach().cpu().numpy()],
        [colors_1.detach().cpu().numpy(), colors_2.detach().cpu().numpy()],
        plot_shape=(1, 2), **kwargs
    )


def vis_knn3(p, p_idx, group_idx_1, group_idx_2, **kwargs):
    group_idx_1 = group_idx_1.long()
    group_idx_2 = group_idx_2.long()
    colors = torch.ones_like(p, dtype=torch.float, device=p.device)
    colors[:] = white
    g_idx_1 = group_idx_1[p_idx].unsqueeze(-1).expand(-1, 3)
    g_idx_2 = group_idx_2[p_idx].unsqueeze(-1).expand(-1, 3)
    g_idx_3 = g_idx_1[torch.isin(g_idx_1, g_idx_2)]
    g_idx_4 = g_idx_1[torch.isin(g_idx_1, g_idx_2, invert=True)]
    colors[g_idx_3] = green  # same
    colors[g_idx_4] = yellow  # diff
    colors[p_idx] = red
    vis_multi_points(
        [p.detach().cpu().numpy()],
        [colors.detach().cpu().numpy()],
        plot_shape=(1, 1), **kwargs
    )

#
# # visual visible total for xyz and xyz_sampled
# if __name__ == '__main__':
#     train_datas = read_gs_features('train')
#     val_datas = read_gs_features('val')
#     train_p0, train_f0, train_means, train_means2d, train_depths, train_visible, train_cov3d, train_conics, train_radius, train_num_tiles_hit = train_datas
#     val_p0, val_f0, val_means, val_means2d, val_depths, val_visible, val_cov3d, val_conics, val_radius, val_num_tiles_hit = val_datas
#
#     n_cameras = 2
#     stride = 64
#     fovy = 120
#
#     gs = NaiveGaussian3D(opt=GaussianOptions(n_cameras=n_cameras, cam_field_size=[512, 512], cam_fovy=fovy, cam_sampler='fps'), device='cuda')
#     uv, depths, visible, camid = gs.projects(train_p0)
#     visible_total = gs.gs_points.visible.sum(-1).squeeze(-1) # [B, N, 1]
#
#
#     def get_cmap(labels):
#         # labels是一个二维数组，是密度图
#         max_pixel = np.max(labels)
#         min_pixel = np.min(labels)
#         delta = max_pixel - min_pixel
#         cmap = ((labels - min_pixel) / delta * 255)
#         # 以下操作是为了反转jet的颜色，不然就会出现数值高的反而是蓝色，数值低的是红色，不像热力图了
#         cmap = cmap * (-1)
#         cmap = cmap + 255
#         cmap = cmap.astype(np.uint8)
#         return cmap
#
#     cmap = get_cmap(visible_total.detach().cpu().numpy())
#     train_color = torch.ones_like(train_p0, device='cuda')
#     visible_color = torch.from_numpy(cv2.applyColorMap(cmap, cv2.COLORMAP_JET)).cuda()
#
#     visible_xyz_sampled, visible_xyz_idx = create_sampler('visible')(xyz=train_p0, visible=visible, n_samples=train_p0.shape[1]//stride)
#     visible_colors_sampled = torch.gather(visible_color, 1, visible_xyz_idx.unsqueeze(-1).expand(-1, -1, 3))
#
#     # gs.apply_sampling(visible_xyz_idx)
#     # xyz_sampled, xyz_idx = create_sampler('fps')(xyz=visible_xyz_sampled, visible=gs.gs_points.visible, n_samples=visible_xyz_sampled.shape[1] // 4)
#     # colors_sampled = torch.gather(visible_colors_sampled, 1, xyz_idx.unsqueeze(-1).expand(-1, -1, 3))
#
#     xyz_sampled, xyz_idx = create_sampler('fps')(xyz=train_p0, visible=visible, n_samples=train_p0.shape[1] // stride)
#     colors_sampled = torch.gather(visible_color, 1, xyz_idx.unsqueeze(-1).expand(-1, -1, 3))
#
#     print(visible_xyz_sampled.shape, xyz_sampled.shape)
#
#     # todo make points and color by func and loop
#
#     vis_multi_points(
#         [
#             train_p0[0].detach().cpu().numpy(), train_p0[0].detach().cpu().numpy(),
#             visible_xyz_sampled[0].detach().cpu().numpy(), xyz_sampled[0].detach().cpu().numpy(),
#             train_p0[1].detach().cpu().numpy(), train_p0[1].detach().cpu().numpy(),
#             visible_xyz_sampled[1].detach().cpu().numpy(), xyz_sampled[1].detach().cpu().numpy(),],
#         [
#             train_color[0].detach().cpu().numpy(), visible_color[0].detach().cpu().numpy(),
#             visible_colors_sampled[0].detach().cpu().numpy(), colors_sampled[0].detach().cpu().numpy(),
#             train_color[1].detach().cpu().numpy(), visible_color[1].detach().cpu().numpy(),
#             visible_colors_sampled[1].detach().cpu().numpy(), colors_sampled[1].detach().cpu().numpy(),],
#         plot_shape=(2, 4), point_size=6)
#
