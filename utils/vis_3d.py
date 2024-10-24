import math
import os

import cv2
import numpy as np
import torch
from einops import repeat
from matplotlib import pyplot as plt
from torch import nn
import matplotlib.markers as mmarkers

from utils import points_scaler


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
    cmap = (labels - min_pixel) / (delta + 1e-6) * 255
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
    """
    Visualize a point cloud with k-nearest neighbors.
        pc is white, center is red, neighbors is blue.
    :param p: point cloud
    :param p_idx: center point index
    :param group_idx: neighbor point index
    :param kwargs:
    :return:
    """
    vis = kwargs.get('vis', True)
    group_idx = group_idx.long()
    colors = torch.zeros_like(p, dtype=torch.float, device=p.device)
    colors[:] = white
    g_idx = group_idx[p_idx]
    colors[g_idx] = blue
    colors[p_idx] = red
    if vis:
        vis_multi_points(
            [p.detach().cpu().numpy()],
            [colors.detach().cpu().numpy()],
            plot_shape=(1, 1), **kwargs
        )
    else:
        return colors


def vis_knn2(p, p_idx, group_idx_1, group_idx_2, **kwargs):
    """
    Visualize a point cloud with k-nearest neighbors.
        pc is white, center is red, neighbors1 is blue, neighbors2 is yellow, same is green.
    :param p: point cloud
    :param p_idx: center point index
    :param group_idx_1: knn1 neighbor point index
    :param group_idx_2: knn2 neighbor point index
    :param kwargs:
    :return:
    """
    vis = kwargs.get('vis', True)
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
    if vis:
        vis_multi_points(
            [p.detach().cpu().numpy(), p.detach().cpu().numpy()],
            [colors_1.detach().cpu().numpy(), colors_2.detach().cpu().numpy()],
            plot_shape=(1, 2), **kwargs
        )
    else:
        return colors_1, colors_2


def vis_knn3(p, p_idx, group_idx_1, group_idx_2, **kwargs):
    """
    Visualize a point cloud with k-nearest neighbors.
        pc is white, center is red, diff is yellow, same is green.
    :param p: point cloud
    :param p_idx: center point index
    :param group_idx_1: knn1 neighbor point index
    :param group_idx_2: knn2 neighbor point index
    :param kwargs:
    :return:
    """
    vis = kwargs.get('vis', True)
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
    if vis:
        vis_multi_points(
            [p.detach().cpu().numpy()],
            [colors.detach().cpu().numpy()],
            plot_shape=(1, 1), **kwargs
        )
    else:
        return colors


def vis_knn4(p, label, p_idx, group_idx_1, group_idx_2, **kwargs):
    """
    Visualize a point cloud with k-nearest neighbors and label.
        pc is white, wrong is yellow, center is red, right is green.
    :param p: point cloud
    :param label: label of points
    :param p_idx: center point index
    :param group_idx_1: knn1 neighbor point index
    :param group_idx_2: knn2 neighbor point index
    :param kwargs:
    :return:
    """
    vis = kwargs.get('vis', True)
    group_idx_1 = group_idx_1.long()
    group_idx_2 = group_idx_2.long()
    colors_1 = torch.zeros_like(p, dtype=torch.float, device=p.device)
    colors_2 = torch.zeros_like(p, dtype=torch.float, device=p.device)
    colors_1[:] = white
    colors_2[:] = white
    g_idx_1 = group_idx_1[p_idx]
    g_idx_2 = group_idx_2[p_idx]
    l = label[p_idx]  # center label
    l_1 = label[g_idx_1] - l  # knn1 labels
    l_2 = label[g_idx_2] - l  # knn2 labels
    l_idx_1 = torch.nonzero(l_1 != 0)  # knn1 wrong index
    l_idx_2 = torch.nonzero(l_2 != 0)  # knn2 wrong index
    l_idx_3 = torch.nonzero(l_1 == 0)  # knn1 right index
    l_idx_4 = torch.nonzero(l_2 == 0)  # knn2 right index
    g_idx_w1 = g_idx_1[l_idx_1].expand(-1, 3)
    g_idx_w2 = g_idx_2[l_idx_2].expand(-1, 3)
    g_idx_r1 = g_idx_1[l_idx_3].expand(-1, 3)
    g_idx_r2 = g_idx_2[l_idx_4].expand(-1, 3)
    colors_1[g_idx_w1] = yellow  # wrong
    colors_2[g_idx_w2] = yellow  # wrong
    colors_1[g_idx_r1] = green  # right
    colors_2[g_idx_r2] = green  # right
    colors_1[p_idx] = red
    colors_2[p_idx] = red

    r1, w1 = g_idx_r1.shape[0], g_idx_w1.shape[0]
    r2, w2 = g_idx_r2.shape[0], g_idx_w2.shape[0]
    # print(f'knn1 right:{r1}, knn1 wrong:{w1}')
    # print(f'knn2 right:{r2}, knn2 wrong:{w2}')
    if vis:
        vis_multi_points(
            [p.detach().cpu().numpy(), p.detach().cpu().numpy()],
            [colors_1.detach().cpu().numpy(), colors_2.detach().cpu().numpy()],
            plot_shape=(1, 2), **kwargs
        )
    else:
        return colors_1, colors_2, r1, w1, r2, w2


def vis_labels(p, label, cmap=None, **kwargs):
    vis = kwargs.get('vis', True)
    # colors = torch.zeros_like(p, dtype=torch.float, device=p.device)
    # u_label = torch.unique(label)
    if cmap is None:
        cmap = calc_cmap(label.detach().cpu().numpy())
    colors = torch.from_numpy(cv2.applyColorMap(cmap, cv2.COLORMAP_RAINBOW)).squeeze()
    if p.is_cuda:
        colors = colors.cuda()
    if vis:
        vis_multi_points([p.detach().cpu().numpy()],
                         [colors.detach().cpu().numpy()],
                         plot_shape=(1, 1), **kwargs)
    else:
        return colors


def vis_projects_3d(p, gs, cam_idx, hidden=False, **kwargs):
    vis = kwargs.get('vis', True)
    p = points_scaler(p.unsqueeze(0), scale=1.0).squeeze(0)
    cameras = gs.gs_points.cameras
    depths = gs.gs_points.depths
    ps = []
    cs = []
    n = int(math.sqrt(len(cam_idx)))
    for i in range(n):
        for j in range(n):
            c_idx = cam_idx[i * n + j]
            visible = depths[:, 0, c_idx] != 0
            camera = cameras[c_idx]
            colors = torch.zeros_like(p)
            colors[:] = white
            if_visible = torch.nonzero(visible)
            if_visible = if_visible.squeeze(-1)
            if hidden:
                visible_p = torch.index_select(p, 0, if_visible)
                visible_c = torch.index_select(colors, 0, if_visible)
            else:
                visible_p = p
                visible_c = colors
                visible_c[if_visible] = yellow

            campos = camera.campos
            target = camera.target
            visible_p = torch.cat([visible_p, campos.unsqueeze(0), target.unsqueeze(0)], dim=0)
            visible_c = torch.cat([visible_c, torch.tensor([[1, 0, 0], [0, 0, 1]], device=p.device)], dim=0)

            ps.append(visible_p.detach().cpu().numpy())
            cs.append(visible_c.detach().cpu().numpy())
    if vis:
        vis_multi_points(ps, cs, plot_shape=(n, n), **kwargs)
    else:
        return cs


def mscatter(x, y, ax=None, m=None, **kw):
    if not ax: ax = plt.gca()
    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def vis_projects_2d(gs, cam_idx, **kwargs):
    vis = kwargs.get('vis', True)
    uv = gs.gs_points.uv
    delta = torch.round((uv % 1) * 1e5) / 1e5
    xy = uv - delta
    colors = gs.gs_points.depths
    if vis:
        n = int(math.sqrt(len(cam_idx)))
        fig, axes = plt.subplots(n, n, dpi=640)
        axes_list = []
        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                axes_list.append(axes[i, j])
        for i in range(len(axes_list)):
            a = axes_list[i]
            c_idx = cam_idx[i]
            c = colors[:, 0, c_idx]
            m = ['o'] * c.shape[0]
            if gs.gs_points.cameras[c_idx].cam_index >= 0:
                m[gs.gs_points.cameras[c_idx].cam_index] = 'v'
            mscatter(xy[:, 0, c_idx], xy[:, 1, c_idx], a, s=4, c=c, m=m, cmap='rainbow')
            a.set_xticks([])
            a.set_yticks([])
            a.set_title(f'camera-{c_idx}', loc='left', fontsize=12)
        fig.set_size_inches(10 * n // 4, 8 * n // 4)
        fig.show()
    else:
        return xy, colors


def vis_visible(p, label, visible, camid, **kwargs):
    vis = kwargs.get('vis', True)
    cmap = calc_cmap(label.detach().cpu().numpy())
    colors = torch.from_numpy(cv2.applyColorMap(cmap, cv2.COLORMAP_RAINBOW)).squeeze()

    i = torch.arange(1, visible.shape[-1]+1)
    i = repeat(i, 'c -> n c', n=visible.shape[0])
    visible_code = (visible * i * camid).mean(dim=-1)
    cmap = calc_cmap(visible_code.detach().cpu().numpy())
    visible_colors = torch.from_numpy(cv2.applyColorMap(cmap, cv2.COLORMAP_RAINBOW))
    visible_colors = visible_colors.squeeze(1)
    if vis:
        vis_multi_points([p.detach().cpu().numpy(), p.detach().cpu().numpy()],
                         [colors.detach().cpu().numpy(), visible_colors.detach().cpu().numpy()],
                         plot_shape=(1, 2), **kwargs)
    else:
        return visible_colors

