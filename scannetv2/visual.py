import __init__

import sys

from utils import read_obj, vis_multi_points

if __name__ == '__main__':
    vis_root = 'visual'
    # 3 4 23 28 33 36 39 42 44 72 73 74 75 76 77 78 94 97 125 136 181 182 206 225 227 228 233 254 281 296
    for idx in range(312):
        rgb = f'{vis_root}/rgb-scannetv2-{idx}.txt'
        gt = f'{vis_root}/gt-scannetv2-{idx}.txt'
        pred = f'{vis_root}/pred-scannetv2-{idx}.txt'

        input_points, input_colors = read_obj(rgb)
        gt_points, gt_colors = read_obj(gt)
        method_points, method_colors = read_obj(pred)
        vis_multi_points([input_points, gt_points, method_points],
                         [input_colors, gt_colors, method_colors],
                         title=f'scannet-{idx}', plot_shape=(1, 3))

