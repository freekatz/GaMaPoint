import __init__

import sys

from utils import read_obj, vis_multi_points

if __name__ == '__main__':
    idx = sys.argv[1]
    vis_root = 'visual'
    rgb = f'{vis_root}/rgb-scannetv2-{idx}.txt'
    gt = f'{vis_root}/gt-scannetv2-{idx}.txt'
    pred = f'{vis_root}/pred-scannetv2-{idx}.txt'

    input_points, input_colors = read_obj(rgb)
    gt_points, gt_colors = read_obj(gt)
    method_points, method_colors = read_obj(pred)
    vis_multi_points([input_points, gt_points, method_points],
                     [input_colors, gt_colors, method_colors],
                     plot_shape=(1, 3))

