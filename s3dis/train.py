import __init__

import argparse
import logging
import os
from glob import glob

import numpy as np
from timm.scheduler.cosine_lr import CosineLRScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from backbone import DelaSemSeg
from backbone.gs_3d import NaiveGaussian3D, merge_gs_list, make_gs_points
from s3dis.configs import model_configs
from s3dis.dataset import S3DIS, s3dis_collate_fn
from utils.ckpt_util import load_state, save_state, cal_model_params
from utils.config import EasyConfig
from utils.logger import setup_logger_dist
from utils.metrics import Metric, AverageMeter
from utils.random import set_random_seed
from utils.timer import Timer


def fix_batch(cfg, gs_list, train=True, warmup=False) -> NaiveGaussian3D:
    if train:
        assert len(gs_list) == cfg.batch_size
    else:
        assert len(gs_list) == 1
    assert gs_list[0].gs_points.p.is_cuda

    if warmup:
        grid_size = cfg.s3dis_warmup_cfg.grid_size
        ks = cfg.s3dis_warmup_cfg.k
    else:
        grid_size = cfg.s3dis_cfg.grid_size
        ks = cfg.s3dis_cfg.k

    for idx in range(len(gs_list)):
        gs = gs_list[idx]
        gs.projects(gs.gs_points.p, cam_seed=idx)
        gs_list[idx].gs_points = make_gs_points(gs.gs_points, grid_size, ks, warmup=warmup)
    gs = merge_gs_list(gs_list)
    return gs


def prepare_exp(cfg):
    exp_root = 'exp'
    exp_id = len(glob(f'{exp_root}/{cfg.exp}-*'))
    cfg.exp_name = f'{cfg.exp}-{exp_id:03d}'
    cfg.exp_dir = f'{exp_root}/{cfg.exp_name}'
    cfg.log_path = f'{cfg.exp_dir}/train.log'
    cfg.best_small_ckpt_path = f'{cfg.exp_dir}/best_small.ckpt'
    cfg.best_ckpt_path = f'{cfg.exp_dir}/best.ckpt'
    cfg.last_ckpt_path = f'{cfg.exp_dir}/last.ckpt'

    os.makedirs(cfg.exp_dir, exist_ok=True)
    cfg.save(f'{cfg.exp_dir}/config.yaml')
    setup_logger_dist(cfg.log_path, 0, name=cfg.exp_name)


def warmup(model: nn.Module, warmup_loader):
    model.train()
    pbar = tqdm(enumerate(warmup_loader), total=warmup_loader.__len__(), desc='Warmup')
    for idx, gs_list in pbar:
        for gs in gs_list:
            gs.gs_points.to_cuda(non_blocking=True)
        gs = fix_batch(cfg, gs_list, train=False, warmup=True)
        gs.gs_points.to_cuda(non_blocking=True)
        target = gs.gs_points.y
        with autocast():
            pred = model(gs)
            loss = F.cross_entropy(pred, target)
        loss.backward()


def train(cfg, model, train_loader, optimizer, scheduler, scaler, epoch, scheduler_steps):
    model.train()
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__(), desc='Train')
    m = Metric(cfg.num_classes)
    loss_meter = AverageMeter()
    for idx, gs_list in pbar:
        for gs in gs_list:
            gs.gs_points.to_cuda(non_blocking=True)
        gs = fix_batch(cfg, gs_list, train=True, warmup=False)
        target = gs.gs_points.y
        with autocast():
            pred = model(gs)
            loss = F.cross_entropy(pred, target)
        if cfg.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step(scheduler_steps)
        scheduler_steps += 1

        m.update(pred, target)
        loss_meter.update(loss.item())
        pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] Loss {loss_meter.avg:.4f}")
    acc, macc, miou, iou = m.calc()
    return loss_meter.avg, miou, macc, iou, acc, scheduler_steps


def validate(cfg, model, val_loader):
    model.eval()
    m = Metric(cfg.num_classes)
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__(), desc='Val')
    for idx, gs_list in pbar:
        for gs in gs_list:
            gs.gs_points.to_cuda(non_blocking=True)
        gs = fix_batch(cfg, gs_list, train=False, warmup=False)
        gs.gs_points.to_cuda(non_blocking=True)
        target = gs.gs_points.y
        with autocast():
            pred = model(gs)
        m.update(pred, target)
    acc, macc, miou, iou = m.calc()
    return miou, macc, iou, acc


def main(cfg):
    set_random_seed(cfg.seed, deterministic=True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.enabled = True

    logging.info(f'cfg:\n{cfg.__str__()}')

    warmup_loader = DataLoader(
        S3DIS(
            dataset_dir=cfg.dataset,
            area=f'!{cfg.val_area}',
            loop=cfg.batch_size,
            train=True,
            warmup=True,
            voxel_max=cfg.s3dis_warmup_cfg.voxel_max,
            batch_size=1,
            gs_opts=cfg.s3dis_warmup_cfg.gs_opts
        ),
        batch_size=1,
        collate_fn=s3dis_collate_fn,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )
    train_loader = DataLoader(
        S3DIS(
            dataset_dir=cfg.dataset,
            area=f'!{cfg.val_area}',
            loop=cfg.train_loop,
            train=True,
            warmup=False,
            voxel_max=cfg.s3dis_cfg.voxel_max,
            batch_size=cfg.batch_size,
            gs_opts=cfg.s3dis_cfg.gs_opts
        ),
        batch_size=cfg.batch_size,
        collate_fn=s3dis_collate_fn,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        S3DIS(
            dataset_dir=cfg.dataset,
            area=cfg.val_area,
            loop=cfg.val_loop,
            train=False,
            warmup=False,
            voxel_max=cfg.s3dis_cfg.voxel_max,
            batch_size=1,
            gs_opts=cfg.s3dis_cfg.gs_opts
        ),
        batch_size=1,
        collate_fn=s3dis_collate_fn,
        pin_memory=True,
        persistent_workers=True,
        num_workers=cfg.num_workers,
    )

    steps_per_epoch = len(train_loader)

    model = DelaSemSeg(cfg.gama_cfg).to('cuda')
    model_size, trainable_model_size = cal_model_params(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    logging.info('Number of trainable params: %.4f M' % (trainable_model_size / 1e6))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.decay)
    scheduler = CosineLRScheduler(optimizer,
                                  t_initial=cfg.epochs * steps_per_epoch,
                                  lr_min=cfg.lr / 10000,
                                  warmup_t=cfg.warmup_epochs * steps_per_epoch,
                                  warmup_lr_init=cfg.lr / 20)
    scaler = GradScaler()

    start_epoch = 1
    best_epoch = 0
    best_miou = 0
    if cfg.mode == 'pretrain':
        model_dict = load_state(cfg.ckpt, model=model, optimizer=optimizer, scaler=scaler)
        start_epoch = model_dict['last_epoch']
        best_epoch = model_dict['best_epoch']
        best_miou = model_dict['best_miou']
        logging.info(f"Loading model from {cfg.ckpt}, best_miou={best_miou}, best_epoch={best_epoch}, start_epoch={start_epoch}")
    cfg.epochs = cfg.epochs + start_epoch
    scheduler_steps = steps_per_epoch * start_epoch

    warmup(model, warmup_loader)

    val_miou, val_macc, val_ious, val_accs = 0., 0., [], []
    macc_when_best = 0.
    writer = SummaryWriter(log_dir=cfg.exp_dir)
    timer = Timer(dec=1)
    timer_meter = AverageMeter()
    for epoch in range(start_epoch, cfg.epochs):
        timer.record(f'epoch_{epoch}_start')
        train_loss, train_miou, train_macc, train_ious, train_accs, scheduler_steps = train(
            cfg, model, train_loader, optimizer, scheduler, scaler, epoch, scheduler_steps,
        )
        lr = optimizer.param_groups[0]['lr']

        is_best = False
        if epoch % cfg.val_freq == 0:
            with torch.no_grad():
                val_miou, val_macc, val_ious, val_accs = validate(
                    cfg, model, val_loader,
                )
            if val_miou > best_miou:
                is_best = True
                best_miou = val_miou
                macc_when_best = val_macc
            with np.printoptions(precision=8, suppress=True):
                logging.info(f'Current ckpt val info: epoch={epoch}'
                             + f'\n\tval_miou={val_miou:.8f} val_macc={val_macc:.8f} val_accs={val_accs.detach().cpu().numpy()}'
                             + f'\n\tval_ious={val_ious.detach().cpu().numpy()}')

        time_cost = timer.record(f'epoch_{epoch - 1}_end')
        timer_meter.update(time_cost)
        logging.info(f'Current ckpt train info: epoch={epoch} lr={lr:.8f}'
                     + f'\n\ttrain_miou={train_miou:.8f} avg_time_cost={timer_meter.avg:.8f}'
                     + f'\n\tval_miou={val_miou:.8f} best_val_miou={best_miou:.8f}')

        if is_best:
            logging.info(f'Find a better ckpt: epoch={epoch}, best_val_miou={best_miou:.8f}')
            best_epoch = epoch
            save_state(cfg.best_small_ckpt_path, model=model)
            save_state(cfg.best_ckpt_path, model=model, optimizer=optimizer, scaler=scaler,
                       best_epoch=best_epoch, last_epoch=epoch + 1, best_miou=best_miou)
        save_state(cfg.last_ckpt_path, model=model, optimizer=optimizer, scaler=scaler,
                   best_epoch=best_epoch, last_epoch=epoch + 1, best_miou=best_miou)
        if writer is not None:
            writer.add_scalar('best_miou', best_miou, epoch)
            writer.add_scalar('val_miou', val_miou, epoch)
            writer.add_scalar('macc_when_best', macc_when_best, epoch)
            writer.add_scalar('val_macc', val_macc, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_miou', train_miou, epoch)
            writer.add_scalar('train_macc', train_macc, epoch)
            writer.add_scalar('lr', lr, epoch)
            writer.add_scalar('time_cost_ms', timer_meter.avg*1000, epoch)
            writer.add_scalar('time_cost_epoch_ms', time_cost*1000, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('s3dis training')
    # for prepare
    parser.add_argument('--exp', type=str, required=False, default='s3dis')
    parser.add_argument('--ckpt', type=str, required=False, default='')
    parser.add_argument('--seed', type=int, required=False, default=np.random.randint(1, 10000))
    parser.add_argument('--model_size', type=str, required=False, default='s',
                        choices=['s', 'b', 'l', 'xl'])

    # for dataset
    parser.add_argument('--dataset', type=str, required=False, default='dataset_link')
    parser.add_argument('--train_loop', type=int, required=False, default=30)
    parser.add_argument('--val_loop', type=int, required=False, default=1)
    parser.add_argument('--val_area', type=str, required=False, default='5')
    parser.add_argument('--batch_size', type=int, required=False, default=8)
    parser.add_argument('--num_workers', type=int, required=False, default=12)

    # for train
    parser.add_argument('--epochs', type=int, required=False, default=100)
    parser.add_argument("--warmup_epochs", type=int, required=False, default=10)
    parser.add_argument("--lr", type=float, required=False, default=6e-3)
    parser.add_argument("--decay", type=float, required=False, default=5e-2)
    parser.add_argument("--ls", type=float, required=False, default=0.2)
    parser.add_argument("--no_amp", action='store_true')

    # for validate
    parser.add_argument('--val_freq', type=int, required=False, default=1)

    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load_args(args)

    s3dis_cfg, s3dis_warmup_cfg, gama_cfg = model_configs[cfg.model_size]
    cfg.s3dis_cfg = s3dis_cfg
    cfg.s3dis_warmup_cfg = s3dis_warmup_cfg
    cfg.gama_cfg = gama_cfg

    ## tmp code
    from types import SimpleNamespace
    dela_args = SimpleNamespace()
    dela_args.ks = cfg.s3dis_cfg.k
    dela_args.depths = [4, 4, 8, 4]
    dela_args.dims = [64, 128, 256, 512]
    dela_args.nbr_dims = [32, 32]
    dela_args.head_dim = 256
    dela_args.num_classes = 13
    drop_path = 0.1
    drop_rates = torch.linspace(0., drop_path, sum(dela_args.depths)).split(dela_args.depths)
    dela_args.drop_paths = [dpr.tolist() for dpr in drop_rates]
    dela_args.head_drops = torch.linspace(0., 0.15, len(dela_args.depths)).tolist()
    dela_args.bn_momentum = 0.02
    dela_args.act = nn.GELU
    dela_args.mlp_ratio = 2
    # gradient checkpoint
    dela_args.use_cp = False

    dela_args.cor_std = [1.6, 3.2, 6.4, 12.8]
    cfg.gama_cfg = dela_args
    ## tmp code

    if cfg.ckpt != '':
        cfg.mode = 'pretrain'
    else:
        cfg.mode = 'train'
    cfg.use_amp = not cfg.no_amp

    # s3dis
    cfg.num_classes = 13
    cfg.ignore_index = None

    prepare_exp(cfg)
    main(cfg)
