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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from backbone import ClsHead, Stage
from scanobjectnn.configs import model_configs
from scanobjectnn.dataset import ScanObjectNN, scanobjectnn_collate_fn
from utils.ckpt_util import load_state, save_state, cal_model_params, resume_state
from utils.config import EasyConfig
from utils.logger import setup_logger_dist, format_dict
from utils.metrics import Timer, Metric, AverageMeter
from utils.random import set_random_seed


def prepare_exp(cfg):
    exp_root = 'exp'
    exp_id = len(glob(f'{exp_root}/{cfg.exp}-*'))
    if cfg.mode == 'resume':
        exp_id -= 1
    cfg.exp_name = f'{cfg.exp}-{exp_id:03d}'
    cfg.exp_dir = f'{exp_root}/{cfg.exp_name}'
    cfg.log_path = f'{cfg.exp_dir}/train.log'
    cfg.best_small_ckpt_path = f'{cfg.exp_dir}/best_small.ckpt'
    cfg.best_ckpt_path = f'{cfg.exp_dir}/best.ckpt'
    cfg.last_ckpt_path = f'{cfg.exp_dir}/last.ckpt'

    os.makedirs(cfg.exp_dir, exist_ok=True)
    cfg.save(f'{cfg.exp_dir}/config.yaml')
    setup_logger_dist(cfg.log_path, 0, name=cfg.exp_name)


def train(cfg, model, train_loader, optimizer, scheduler, scaler, epoch, scheduler_steps):
    model.train()
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__(), desc='Train')
    m = Metric(cfg.num_classes)
    loss_meter = AverageMeter()
    diff_meter = AverageMeter()
    steps_per_epoch = len(train_loader)
    for idx, gs in pbar:
        lam = scheduler_steps/(epoch*steps_per_epoch)
        lam = 3e-3 ** lam * 0.25
        scheduler.step(scheduler_steps)
        scheduler_steps += 1
        gs.gs_points.to_cuda(non_blocking=True)
        target = gs.gs_points.y
        with autocast():
            pred, diff = model(gs)
            loss = F.cross_entropy(pred, target, label_smoothing=cfg.ls, ignore_index=cfg.ignore_index)
        optimizer.zero_grad(set_to_none=True)
        if cfg.use_amp:
            scaler.scale(loss + diff*lam).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = loss + diff*lam
            loss.backward()
            optimizer.step()

        m.update(pred, target)
        loss_meter.update(loss.item())
        diff_meter.update(diff.item())
        pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                             + f"Loss {loss_meter.avg:.4f} "
                             + f"Diff {diff_meter.avg:.4f} "
                             + f"mACC {m.calc_macc():.4f}")
    acc, macc, miou, iou = m.calc()
    return loss_meter.avg, diff_meter.avg, miou, macc, iou, acc, scheduler_steps


def validate(cfg, model, val_loader, epoch):
    model.eval()
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__(), desc='Val')
    m = Metric(cfg.num_classes)
    loss_meter = AverageMeter()
    for idx, gs in pbar:
        gs.gs_points.to_cuda(non_blocking=True)
        target = gs.gs_points.y
        with autocast():
            pred = model(gs)
            loss = F.cross_entropy(pred, target, label_smoothing=cfg.ls, ignore_index=cfg.ignore_index)
        m.update(pred, target)
        loss_meter.update(loss.item())
        pbar.set_description(f"Val Epoch [{epoch}/{cfg.epochs}] "
                             + f"Loss {loss_meter.avg:.4f} "
                             + f"mACC {m.calc_macc():.4f}")
    acc, macc, miou, iou = m.calc()
    return loss_meter.avg, miou, macc, iou, acc


def main(cfg):
    set_random_seed(cfg.seed, deterministic=True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.enabled = True

    logging.info(f'Config:\n{cfg.__str__()}')

    train_loader = DataLoader(
        ScanObjectNN(
            dataset_dir=cfg.dataset,
            train=True,
            warmup=False,
            num_points=cfg.scanobjectnn_cfg.num_points,
            k=cfg.scanobjectnn_cfg.k,
            k_gs=cfg.scanobjectnn_cfg.k_gs,
            strides=cfg.scanobjectnn_cfg.strides,
            visible_sample_stride=cfg.scanobjectnn_cfg.visible_sample_stride,
            batch_size=cfg.batch_size,
            gs_opts=cfg.scanobjectnn_cfg.gs_opts
        ),
        batch_size=cfg.batch_size,
        collate_fn=scanobjectnn_collate_fn,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        ScanObjectNN(
            dataset_dir=cfg.dataset,
            train=False,
            warmup=False,
            num_points=cfg.scanobjectnn_cfg.num_points,
            k=cfg.scanobjectnn_cfg.k,
            k_gs=cfg.scanobjectnn_cfg.k_gs,
            strides=cfg.scanobjectnn_cfg.strides,
            visible_sample_stride=cfg.scanobjectnn_cfg.visible_sample_stride,
            batch_size=cfg.batch_size,
            gs_opts=cfg.scanobjectnn_cfg.gs_opts
        ),
        batch_size=cfg.batch_size,
        collate_fn=scanobjectnn_collate_fn,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        num_workers=cfg.num_workers,
    )

    stage = Stage(
        **cfg.gama_cfg.stage_cfg,
        task_type='cls',
    ).to('cuda')
    model = ClsHead(
        stage=stage,
        num_classes=cfg.gama_cfg.num_classes,
        bn_momentum=cfg.gama_cfg.bn_momentum,
    ).to('cuda')
    model_size, trainable_model_size = cal_model_params(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    logging.info('Number of trainable params: %.4f M' % (trainable_model_size / 1e6))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.decay)
    scaler = GradScaler()
    start_epoch = 1
    best_epoch = 0
    best_accs = 0
    if cfg.mode == 'resume':
        cfg.ckpt = cfg.last_ckpt_path
        model_dict = resume_state(model, cfg.ckpt, optimizer=optimizer, scaler=scaler)
        start_epoch = model_dict['last_epoch'] + 1
        best_epoch = model_dict['best_epoch']
        best_accs = model_dict['best_accs']
        logging.info(f"Resume model from {cfg.ckpt}, best_accs={best_accs:.4f}, best_epoch={best_epoch}, start_epoch={start_epoch}")
    if cfg.mode == 'finetune':
        model_dict = load_state(model, cfg.ckpt, optimizer=optimizer, scaler=scaler)
        best_epoch = model_dict['best_epoch']
        best_accs = model_dict['best_accs']
        logging.info(f"Finetune model from {cfg.ckpt}, best_accs={best_accs:.4f}, best_epoch={best_epoch}, start_epoch={start_epoch}")

    steps_per_epoch = len(train_loader)
    scheduler_steps = steps_per_epoch * (start_epoch - 1)
    scheduler = CosineLRScheduler(optimizer,
                                  t_initial=cfg.epochs * steps_per_epoch,
                                  lr_min=cfg.lr / 10000,
                                  cycle_decay=cfg.lr_decay,
                                  warmup_t=cfg.warmup_epochs * steps_per_epoch,
                                  warmup_lr_init=cfg.lr / 20)

    val_macc, val_accs = 0., 0.
    macc_when_best = 0.
    writer = SummaryWriter(log_dir=cfg.exp_dir)
    timer = Timer(dec=1)
    timer_meter = AverageMeter()

    for epoch in range(start_epoch, cfg.epochs + 1):
        timer.record(f'E{epoch}_start')
        train_loss, train_diff, _, train_macc, _, train_accs, scheduler_steps = train(
            cfg, model, train_loader, optimizer, scheduler, scaler, epoch, scheduler_steps,
        )
        lr = optimizer.param_groups[0]['lr']
        time_cost = timer.record(f'epoch_{epoch}_end')
        timer_meter.update(time_cost)
        logging.info(
            f'@E{epoch} train:    macc={train_macc:.4f} oa={train_accs:.4f}')

        is_best = False
        if epoch % cfg.val_freq == 0:
            with torch.no_grad():
                val_loss, _, val_macc, _, val_accs = validate(
                    cfg, model, val_loader, epoch,
                )
            logging.info(f'@E{epoch} val:      macc={val_macc:.4f} oa={val_accs:.4f}')
            if val_accs > best_accs:
                logging.info(f'@E{epoch} new best: oa {val_accs:.4f} => {best_accs:.4f}')
                is_best = True
                best_accs = val_accs
                macc_when_best = val_macc
            else:
                logging.info(f'@E{epoch} cur best: oa {best_accs:.4f}')
        if is_best:
            train_info = {
                'macc': train_macc,
                'oa': train_accs,
                'loss': train_loss,
                'diff': train_diff,
                'lr': f"{lr:.6f}",
                'time_cost': f"{time_cost:.2f}s",
                'time_cost_avg': f"{timer_meter.avg:.2f}s",
            }
            val_info = {
                'macc': val_macc,
                'oa': val_accs,
                'loss': val_loss,
            }
            logging.info(f'@E{epoch} summary:'
                         + f'\ntrain: \n{format_dict(train_info)}'
                         + f'\nval: \n{format_dict(val_info)}')
            best_epoch = epoch
            save_state(cfg.best_small_ckpt_path, model=model)
            save_state(cfg.best_ckpt_path, model=model, optimizer=optimizer, scaler=scaler,
                       best_epoch=best_epoch, last_epoch=epoch, best_accs=best_accs)
        save_state(cfg.last_ckpt_path, model=model, optimizer=optimizer, scaler=scaler,
                   best_epoch=best_epoch, last_epoch=epoch, best_accs=best_accs)
        if writer is not None:
            writer.add_scalar('best_accs', best_accs, epoch)
            writer.add_scalar('val_accs', val_accs, epoch)
            writer.add_scalar('macc_when_best', macc_when_best, epoch)
            writer.add_scalar('val_macc', val_macc, epoch)
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_diff', train_diff, epoch)
            writer.add_scalar('train_accs', train_accs, epoch)
            writer.add_scalar('train_macc', train_macc, epoch)
            writer.add_scalar('lr', lr, epoch)
            writer.add_scalar('time_cost_avg', timer_meter.avg, epoch)
            writer.add_scalar('time_cost', time_cost, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('scanobjectnn training')
    # for prepare
    parser.add_argument('--exp', type=str, required=False, default='scanobjectnn')
    parser.add_argument('--mode', type=str, required=False, default='train', choices=['train', 'finetune', 'resume'])
    parser.add_argument('--ckpt', type=str, required=False, default='')
    parser.add_argument('--seed', type=int, required=False, default=np.random.randint(1, 10000))
    parser.add_argument('--model_size', type=str, required=False, default='s',
                        choices=['s', 'b', 'l', 'xl'])

    # for dataset
    parser.add_argument('--dataset', type=str, required=False, default='dataset_link')
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--num_workers', type=int, required=False, default=12)

    # for train
    parser.add_argument('--epochs', type=int, required=False, default=300)
    parser.add_argument("--warmup_epochs", type=int, required=False, default=30)
    parser.add_argument("--lr", type=float, required=False, default=1e-3)
    parser.add_argument("--lr_decay", type=float, required=False, default=1.-1e-3)
    parser.add_argument("--decay", type=float, required=False, default=0.05)
    parser.add_argument("--ls", type=float, required=False, default=0.2)
    parser.add_argument("--no_amp", action='store_true')

    # for validate
    parser.add_argument('--val_freq', type=int, required=False, default=1)

    # for model
    parser.add_argument("--use_cp", action='store_true')

    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load_args(args)

    scanobjectnn_cfg, scanobjectnn_warmup_cfg, gama_cfg = model_configs[cfg.model_size]
    cfg.scanobjectnn_cfg = scanobjectnn_cfg
    cfg.scanobjectnn_warmup_cfg = scanobjectnn_warmup_cfg
    cfg.gama_cfg = gama_cfg
    cfg.gama_cfg.stage_cfg.use_cp = cfg.use_cp
    if cfg.use_cp:
        cfg.gama_cfg.stage_cfg.bn_momentum = 1 - (1 - cfg.gama_cfg.bn_momentum) ** 0.5

    if cfg.mode == 'finetune':
        assert cfg.ckpt != ''
    cfg.use_amp = not cfg.no_amp

    # scanobjectnn
    cfg.num_classes = 15
    cfg.ignore_index = -100

    prepare_exp(cfg)
    main(cfg)
