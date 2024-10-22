import __init__

import argparse
import logging
import os
from glob import glob
import sys

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from backbone import Backbone, SegPartHead, merge_gs_list
from shapenetpart.configs import model_configs
from shapenetpart.dataset import ShapeNetPartNormalTest, shapenetpart_collate_fn, get_ins_mious, ShapeNetPartNormal
from utils import EasyConfig, setup_logger_dist, set_random_seed, resume_state, Timer, AverageMeter, Metric, \
    cal_model_params, cal_model_flops
from utils.logger import format_dict, format_list


def prepare_exp(cfg):
    exp_root = 'exp-test'
    exp_id = len(glob(f'{exp_root}/{cfg.exp}-*'))
    cfg.exp_name = f'{cfg.exp}-{exp_id:03d}'
    cfg.exp_dir = f'{exp_root}/{cfg.exp_name}'
    cfg.log_path = f'{cfg.exp_dir}/test.log'

    os.makedirs(cfg.exp_dir, exist_ok=True)
    setup_logger_dist(cfg.log_path, 0, name=cfg.exp_name)
    logfile = open(cfg.log_path, "a", 1)
    sys.stdout = logfile


def cal_flops(cfg, model):
    cfg.model_cfg.train_cfg.voxel_max = 1024
    cfg.model_cfg.train_cfg.n_samples = [1024, 256, 64]
    ds = ShapeNetPartNormal(
        dataset_dir=cfg.dataset,
        train=True,
        voxel_max=cfg.model_cfg.train_cfg.voxel_max,
        k=cfg.model_cfg.train_cfg.k,
        n_samples=cfg.model_cfg.train_cfg.n_samples,
        alpha=cfg.model_cfg.train_cfg.alpha,
        batch_size=1,
        gs_opts=cfg.model_cfg.train_cfg.gs_opts
    )
    gs = merge_gs_list([ds[0][0]])
    gs.gs_points.to_cuda(non_blocking=True)
    cal_model_flops(model, inputs=dict(gs=gs, shape=torch.tensor([ds[0][1]], device=gs.device)))


@torch.no_grad()
def main(cfg):
    torch.cuda.set_device(0)
    set_random_seed(cfg.seed, deterministic=True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.enabled = True

    logging.info(f'Config:\n{cfg.__str__()}')

    test_ds = ShapeNetPartNormalTest(
        presample_path=cfg.presample_path,
        k=cfg.model_cfg.train_cfg.k,
        n_samples=cfg.model_cfg.train_cfg.n_samples,
        alpha=cfg.model_cfg.train_cfg.alpha,
        batch_size=cfg.batch_size,
        gs_opts=cfg.model_cfg.train_cfg.gs_opts
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        collate_fn=shapenetpart_collate_fn,
        pin_memory=True,
        persistent_workers=True,
        drop_last = True,
        num_workers=cfg.num_workers,
    )
    backbone = Backbone(
        **cfg.model_cfg.backbone_cfg,
        task_type='segpart',
    ).to('cuda')
    model = SegPartHead(
        backbone=backbone,
        num_classes=cfg.model_cfg.num_classes,
        shape_classes=cfg.shape_classes,
        bn_momentum=cfg.model_cfg.bn_momentum,
    ).to('cuda')
    model_size, trainable_model_size = cal_model_params(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    logging.info('Number of trainable params: %.4f M' % (trainable_model_size / 1e6))

    model.eval()
    cal_flops(cfg, model)

    if cfg.ckpt == '':
        return
    resume_state(model, cfg.ckpt, compat=True)

    writer = SummaryWriter(log_dir=cfg.exp_dir)
    timer = Timer(dec=1)
    timer_meter = AverageMeter()
    pbar = tqdm(enumerate(test_loader), total=test_loader.__len__(), desc='Testing')
    steps_per_epoch = len(test_loader)
    cls_mious = torch.zeros(cfg.shape_classes, dtype=torch.float32, device="cuda")
    cls_nums = torch.zeros(cfg.shape_classes, dtype=torch.int32, device="cuda")
    ins_miou_list = []
    for idx, gs in pbar:
        gs.gs_points.to_cuda(non_blocking=True)
        shape = gs.gs_points.__get_attr__('shape')
        target = gs.gs_points.y
        B, N = cfg.batch_size, target.shape[0] // cfg.batch_size
        timer.record(f'I{idx}_start')
        with autocast():
            pred = model(gs, shape)
            pred = pred.max(dim=1)[1].view(B, N)
            target = target.view(B, N)
        time_cost = timer.record(f'I{idx}_end')
        timer_meter.update(time_cost)
        ins_mious = get_ins_mious(pred, target, shape, test_ds.class2parts)
        ins_miou_list += ins_mious
        for shape_idx in range(shape.shape[0]):  # sample_idx
            gt_label = int(shape[shape_idx].cpu().numpy())
            # add the iou belongs to this cat
            cls_mious[gt_label] += ins_mious[shape_idx]
            cls_nums[gt_label] += 1
        pbar.set_description(f"Testing [{idx}/{steps_per_epoch}]")
        if writer is not None and idx % cfg.metric_freq == 0:
            writer.add_scalar('time_cost_avg', timer_meter.avg, idx)
            writer.add_scalar('time_cost', time_cost, idx)
    ins_mious_sum, count = torch.sum(torch.stack(ins_miou_list)), torch.tensor(len(ins_miou_list)).cuda()
    for cat_idx in range(16):
        # indicating this cat is included during previous iou appending
        if cls_nums[cat_idx] > 0:
            cls_mious[cat_idx] = cls_mious[cat_idx] / cls_nums[cat_idx]

    ins_miou = ins_mious_sum / count
    cls_miou = torch.mean(cls_mious)
    cls_mious = [round(cm, 2) for cm in cls_mious.tolist()]
    test_info = {
                'ins_miou': ins_miou,
                'cls_miou': cls_miou,
            }
    logging.info(f'Summary:'
                 + f'\nval: \n{format_dict(test_info)}'
                 + f'\nious: \n{format_list(ShapeNetPartNormalTest.get_classes(), cls_mious)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('shapenetpart testing')
    # for prepare
    parser.add_argument('--exp', type=str, required=False, default='shapenetpart')
    parser.add_argument('--mode', type=str, required=False, default='train', choices=['test', 'val'])
    parser.add_argument('--ckpt', type=str, required=False, default='')
    parser.add_argument('--seed', type=int, required=False, default=np.random.randint(1, 10000))
    parser.add_argument('--model_size', type=str, required=False, default='s',
                        choices=['s', 'l', 'xl', 'c'])

    # for dataset
    parser.add_argument('--dataset', type=str, required=False, default='dataset_link')
    parser.add_argument('--presample', type=str, required=False, default='shapenetpart_presample.pt')
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--num_workers', type=int, required=False, default=12)

    # for test
    parser.add_argument("--metric_freq", type=int, required=False, default=1)

    # for model
    parser.add_argument("--use_cp", action='store_true')

    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load_args(args)

    model_cfg = model_configs[cfg.model_size]
    cfg.model_cfg = model_cfg
    cfg.model_cfg.backbone_cfg.use_cp = cfg.use_cp
    if cfg.use_cp:
        cfg.model_cfg.backbone_cfg.bn_momentum = 1 - (1 - cfg.model_cfg.bn_momentum) ** 0.5
    cfg.presample_path = os.path.join(cfg.dataset, cfg.presample)

    # shapenetpart
    cfg.num_classes = 50
    cfg.shape_classes = 16
    cfg.ignore_index = 255

    prepare_exp(cfg)
    main(cfg)
