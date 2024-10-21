import math

import __init__

import argparse
import logging
import os
from glob import glob

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from backbone import Backbone, SegSemHead
from scannetv2.configs import model_configs
from scannetv2.dataset import ScanNetV2, scannetv2_collate_fn
from utils import EasyConfig, setup_logger_dist, set_random_seed, resume_state, Timer, AverageMeter, Metric, \
    cal_model_params, write_obj
from utils.logger import format_dict, format_list


def prepare_exp(cfg):
    exp_root = 'exp-test'
    exp_id = len(glob(f'{exp_root}/{cfg.exp}-*'))
    cfg.exp_name = f'{cfg.exp}-{exp_id:03d}'
    cfg.exp_dir = f'{exp_root}/{cfg.exp_name}'
    cfg.log_path = f'{cfg.exp_dir}/test.log'
    cfg.vis_root = 'visual'

    os.makedirs(cfg.exp_dir, exist_ok=True)
    os.makedirs(cfg.vis_root, exist_ok=True)
    setup_logger_dist(cfg.log_path, 0, name=cfg.exp_name)


def save_vis_results(cfg, file_name, xyz, feat, label, pred):
    rgb = feat.cpu().numpy().squeeze() * 250.0 / 255.0

    label = label.long()
    gt = label.cpu().numpy().squeeze()
    gt = cfg.cmap[gt, :]

    pred = pred.argmax(1)
    pred = pred.cpu().numpy().squeeze()
    pred = cfg.cmap[pred, :]

    write_obj(xyz, rgb, f'{cfg.vis_root}/rgb-{file_name}.txt')
    # output ground truth labels
    write_obj(xyz, gt, f'{cfg.vis_root}/gt-{file_name}.txt')
    # output pred labels
    write_obj(xyz, pred,  f'{cfg.vis_root}/pred-{file_name}.txt')


@torch.no_grad()
def main(cfg):
    torch.cuda.set_device(0)
    set_random_seed(cfg.seed, deterministic=True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.enabled = True

    logging.info(f'Config:\n{cfg.__str__()}')

    test_ds = ScanNetV2(
            dataset_dir=cfg.dataset,
            loop=cfg.test_loop,
            train=False,
            warmup=False,
            voxel_max=cfg.model_cfg.train_cfg.voxel_max,
            k=cfg.model_cfg.train_cfg.k,
            grid_size=cfg.model_cfg.train_cfg.grid_size,
            alpha=cfg.model_cfg.train_cfg.alpha,
            batch_size=1,
            gs_opts=cfg.model_cfg.train_cfg.gs_opts
        )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        collate_fn=scannetv2_collate_fn,
        pin_memory=True,
        persistent_workers=True,
        num_workers=cfg.num_workers,
    )
    cmap = {}
    for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
        cmap[i] = test_ds.label2color[x]
    cfg.cmap = np.array([*cmap.values()]).astype(np.float32) / 255.

    backbone = Backbone(
        **cfg.model_cfg.backbone_cfg,
    ).to('cuda')
    model = SegSemHead(
        backbone=backbone,
        num_classes=cfg.model_cfg.num_classes,
        bn_momentum=cfg.model_cfg.bn_momentum,
    ).to('cuda')
    model_size, trainable_model_size = cal_model_params(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    logging.info('Number of trainable params: %.4f M' % (trainable_model_size / 1e6))

    resume_state(model, cfg.ckpt, compat=True)
    model.eval()

    writer = SummaryWriter(log_dir=cfg.exp_dir)
    timer = Timer(dec=1)
    timer_meter = AverageMeter()
    m = Metric(cfg.num_classes)
    pbar = tqdm(enumerate(test_loader), total=test_loader.__len__(), desc='Testing')
    steps_per_epoch = len(test_loader)
    for idx, gs in pbar:
        gs.gs_points.to_cuda(non_blocking=True)
        target = gs.gs_points.y
        mask = target != cfg.ignore_index
        timer.record(f'I{idx}_start')
        with autocast():
            pred = model(gs)
        time_cost = timer.record(f'I{idx}_end')
        timer_meter.update(time_cost)
        m.update(pred[mask], target[mask])
        pbar.set_description(f"Testing [{idx}/{steps_per_epoch}] "
                             + f"mACC {m.calc_macc():.4f}")
        if writer is not None and idx % cfg.metric_freq == 0:
            writer.add_scalar('time_cost_avg', timer_meter.avg, idx)
            writer.add_scalar('time_cost', time_cost, idx)
        if cfg.vis:
            save_vis_results(
                cfg,
                f'scannetv2-{idx}',
                gs.gs_points.p[mask],
                gs.gs_points.f[:, :3][mask],
                target[mask],
                pred[mask],
            )
    acc, macc, miou, iou = m.calc()
    test_info = {
        'miou': miou,
        'macc': macc,
        'oa': acc,
        'time_cost_avg': f"{timer_meter.avg:.2f}s",
    }
    logging.info(f'Summary:'
                 + f'\ntest: \n{format_dict(test_info)}'
                 + f'\nious: \n{format_list(ScanNetV2.get_classes(), iou)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('scannetv2 testing')
    # for prepare
    parser.add_argument('--exp', type=str, required=False, default='scannetv2')
    parser.add_argument('--mode', type=str, required=False, default='train', choices=['test', 'val'])
    parser.add_argument('--ckpt', type=str, required=False, default='')
    parser.add_argument('--seed', type=int, required=False, default=np.random.randint(1, 10000))
    parser.add_argument('--model_size', type=str, required=False, default='s',
                        choices=['s', 'l', 'xl', 'c'])

    # for dataset
    parser.add_argument('--dataset', type=str, required=False, default='dataset_link')
    parser.add_argument('--test_loop', type=int, required=False, default=1)
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--num_workers', type=int, required=False, default=12)

    # for test
    parser.add_argument("--metric_freq", type=int, required=False, default=1)

    # for vis
    parser.add_argument("--vis", action='store_true')

    # for model
    parser.add_argument("--use_cp", action='store_true')

    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load_args(args)
    assert cfg.ckpt != ''

    model_cfg = model_configs[cfg.model_size]
    cfg.model_cfg = model_cfg
    cfg.model_cfg.backbone_cfg.use_cp = cfg.use_cp
    if cfg.use_cp:
        cfg.model_cfg.backbone_cfg.bn_momentum = 1 - (1 - cfg.model_cfg.bn_momentum) ** 0.5

    # scannetv2
    cfg.num_classes = 20
    cfg.ignore_index = 20

    prepare_exp(cfg)
    main(cfg)
