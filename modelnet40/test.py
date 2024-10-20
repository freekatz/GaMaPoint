import __init__

import logging
import argparse
from glob import glob
import os

from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import torch

from backbone import Backbone, ClsHead
from modelnet40.configs import model_configs
from modelnet40.dataset import ModelNet40, modelnet40_collate_fn
from utils import EasyConfig, setup_logger_dist, set_random_seed, resume_state, Timer, AverageMeter, Metric, \
    cal_model_params
from utils.logger import format_dict


def prepare_exp(cfg):
    exp_root = 'exp-test'
    exp_id = len(glob(f'{exp_root}/{cfg.exp}-*'))
    cfg.exp_name = f'{cfg.exp}-{exp_id:03d}'
    cfg.exp_dir = f'{exp_root}/{cfg.exp_name}'
    cfg.log_path = f'{cfg.exp_dir}/test.log'

    os.makedirs(cfg.exp_dir, exist_ok=True)
    setup_logger_dist(cfg.log_path, 0, name=cfg.exp_name)


def main(cfg):
    torch.cuda.set_device(0)
    set_random_seed(cfg.seed, deterministic=True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.enabled = True

    logging.info(f'Config:\n{cfg.__str__()}')

    test_loader = DataLoader(
        ModelNet40(
            dataset_dir=cfg.dataset,
            train=False,
            warmup=False,
            num_points=cfg.model_cfg.train_cfg.num_points,
            k=cfg.model_cfg.train_cfg.k,
            n_samples=cfg.model_cfg.train_cfg.n_samples,
            visible_sample_stride=cfg.model_cfg.train_cfg.visible_sample_stride,
            alpha=cfg.model_cfg.train_cfg.alpha,
            batch_size=cfg.batch_size,
            gs_opts=cfg.model_cfg.train_cfg.gs_opts
        ),
        batch_size=cfg.batch_size,
        collate_fn=modelnet40_collate_fn,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        num_workers=cfg.num_workers,
    )

    backbone = Backbone(
        **cfg.model_cfg.backbone_cfg,
        task_type='cls',
    ).to('cuda')
    model = ClsHead(
        backbone=backbone,
        num_classes=cfg.model_cfg.num_classes,
        bn_momentum=cfg.model_cfg.bn_momentum,
        cls_type='max',
    ).to('cuda')
    model_size, trainable_model_size = cal_model_params(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    logging.info('Number of trainable params: %.4f M' % (trainable_model_size / 1e6))

    resume_state(model, cfg.ckpt)
    model.eval()

    writer = SummaryWriter(log_dir=cfg.exp_dir)
    timer = Timer(dec=1)
    timer_meter = AverageMeter()
    m = Metric(cfg.num_classes)
    pbar = tqdm(enumerate(test_loader), total=test_loader.__len__(), desc='Test')
    steps_per_epoch = len(test_loader)
    for idx, gs in pbar:
        gs.gs_points.to_cuda(non_blocking=True)
        target = gs.gs_points.y
        timer.record(f'I{idx}_start')
        with autocast():
            pred = model(gs)
        time_cost = timer.record(f'I{idx}_end')
        if idx == 0:
            time_cost = 0
        timer_meter.update(time_cost)
        m.update(pred, target)
        pbar.set_description(f"Testing [{idx}/{steps_per_epoch}] "
                             + f"mACC {m.calc_macc():.4f}")
        if writer is not None and idx % cfg.metric_freq == 0:
            writer.add_scalar('time_cost_avg', timer_meter.avg, idx)
            writer.add_scalar('time_cost', time_cost, idx)
    acc, macc, miou, iou = m.calc()
    test_info = {
        'macc': macc,
        'oa': acc,
        'time_cost_avg': f"{timer_meter.avg:.2f}s",
    }
    logging.info(f'Summary:'
                 + f'\n{format_dict(test_info)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('modelnet40 testing')
    # for prepare
    parser.add_argument('--exp', type=str, required=False, default='modelnet40')
    parser.add_argument('--mode', type=str, required=False, default='train', choices=['test', 'val'])
    parser.add_argument('--ckpt', type=str, required=False, default='')
    parser.add_argument('--seed', type=int, required=False, default=np.random.randint(1, 10000))
    parser.add_argument('--model_size', type=str, required=False, default='s',
                        choices=['s', 'l', 'xl', 'c'])

    # for dataset
    parser.add_argument('--dataset', type=str, required=False, default='dataset_link')
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--num_workers', type=int, required=False, default=12)

    # for test
    parser.add_argument("--metric_freq", type=int, required=False, default=1)

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

    # modelnet40
    cfg.num_classes = 40

    prepare_exp(cfg)
    main(cfg)
