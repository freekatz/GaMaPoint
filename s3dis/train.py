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

from backbone import SegHead, Encoder, Decoder
from s3dis.configs import model_configs
from s3dis.dataset import S3DIS, s3dis_collate_fn
from utils.ckpt_util import load_state, save_state, cal_model_params, resume_state
from utils.config import EasyConfig
from utils.logger import setup_logger_dist
from utils.metrics import Metric, AverageMeter
from utils.random import set_random_seed
from utils.timer import Timer


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


def warmup(model: nn.Module, warmup_loader):
    model.train()
    pbar = tqdm(enumerate(warmup_loader), total=warmup_loader.__len__(), desc='Warmup')
    for idx, gs in pbar:
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
    for idx, gs in pbar:
        gs.gs_points.to_cuda(non_blocking=True)
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
        pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] Loss {loss_meter.avg:.4f} mACC {m.calc_macc():.4f}")
    acc, macc, miou, iou = m.calc()
    return loss_meter.avg, miou, macc, iou, acc, scheduler_steps


def validate(cfg, model, val_loader, epoch):
    model.eval()
    m = Metric(cfg.num_classes)
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__(), desc='Val')
    for idx, gs in pbar:
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

    logging.info(f'Config:\n{cfg.__str__()}')

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

    encoder = Encoder(
        **cfg.gama_cfg.encoder_cfg,
    ).to('cuda')
    decoder = Decoder(
        **cfg.gama_cfg.decoder_cfg,
    ).to('cuda')
    model = SegHead(
        encoder=encoder,
        decoder=decoder,
        num_classes=cfg.gama_cfg.num_classes,
        bn_momentum=cfg.gama_cfg.bn_momentum,
    ).to('cuda')
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
    if cfg.mode == 'resume':
        cfg.ckpt = cfg.last_ckpt_path
        model_dict = resume_state(model, cfg.ckpt, optimizer=optimizer, scaler=scaler)
        start_epoch = model_dict['last_epoch'] + 1
        best_epoch = model_dict['best_epoch']
        best_miou = model_dict['best_miou']
        cfg.epochs = cfg.epochs + start_epoch - 1
        logging.info(f"Resume model from {cfg.ckpt}, best_miou={best_miou:.4f}, best_epoch={best_epoch}, start_epoch={start_epoch}")
    if cfg.mode == 'finetune':
        model_dict = load_state(model, cfg.ckpt, optimizer=optimizer, scaler=scaler)
        best_epoch = model_dict['best_epoch']
        best_miou = model_dict['best_miou']
        logging.info(f"Finetune model from {cfg.ckpt}, best_miou={best_miou:.4f}, best_epoch={best_epoch}, start_epoch={start_epoch}")
    scheduler_steps = steps_per_epoch * start_epoch

    warmup(model, warmup_loader)

    val_miou, val_macc, val_ious, val_accs = 0., 0., [], []
    macc_when_best = 0.
    writer = SummaryWriter(log_dir=cfg.exp_dir)
    timer = Timer(dec=1)
    timer_meter = AverageMeter()
    for epoch in range(start_epoch, cfg.epochs + 1):
        timer.record(f'E{epoch}_start')
        train_loss, train_miou, train_macc, train_ious, train_accs, scheduler_steps = train(
            cfg, model, train_loader, optimizer, scheduler, scaler, epoch, scheduler_steps,
        )
        lr = optimizer.param_groups[0]['lr']
        time_cost = timer.record(f'epoch_{epoch - 1}_end')
        timer_meter.update(time_cost)
        logging.info(f'@E{epoch} train results: '
                     + f'\nlr={lr:.6f} train_loss={train_loss:.4f} '
                     + f'train_macc={train_miou:.4f} train_accs={train_accs:.4f} train_miou={train_miou:.4f} '
                     + f'time_cost={timer_meter.avg:.6f}s avg_time_cost={timer_meter.avg:.6f}s')

        is_best = False
        if epoch % cfg.val_freq == 0:
            with torch.no_grad():
                val_miou, val_macc, val_ious, val_accs = validate(
                    cfg, model, val_loader, epoch,
                )
            if val_miou > best_miou:
                is_best = True
                best_miou = val_miou
                macc_when_best = val_macc
            with np.printoptions(precision=4, suppress=True):
                logging.info(f'@E{epoch} val results: '
                             + f'\nval_macc={val_macc:.4f} val_accs={val_accs.detach().cpu().numpy():.4f} '
                             + f'val_miou={val_miou:.4f}  best_val_miou={best_miou:.4f}'
                             + f'\nval_ious={val_ious.detach().cpu().numpy()}')
        if is_best:
            logging.info(f'@E{epoch} new best: best_val_miou={best_miou:.4f}')
            best_epoch = epoch
            save_state(cfg.best_small_ckpt_path, model=model)
            save_state(cfg.best_ckpt_path, model=model, optimizer=optimizer, scaler=scaler,
                       best_epoch=best_epoch, last_epoch=epoch, best_miou=best_miou)
        save_state(cfg.last_ckpt_path, model=model, optimizer=optimizer, scaler=scaler,
                   best_epoch=best_epoch, last_epoch=epoch, best_miou=best_miou)
        if writer is not None:
            writer.add_scalar('best_miou', best_miou, epoch)
            writer.add_scalar('val_miou', val_miou, epoch)
            writer.add_scalar('macc_when_best', macc_when_best, epoch)
            writer.add_scalar('val_macc', val_macc, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_miou', train_miou, epoch)
            writer.add_scalar('train_macc', train_macc, epoch)
            writer.add_scalar('lr', lr, epoch)
            writer.add_scalar('time_cost', timer_meter.avg, epoch)
            writer.add_scalar('time_cost_epoch', time_cost, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('s3dis training')
    # for prepare
    parser.add_argument('--exp', type=str, required=False, default='s3dis')
    parser.add_argument('--mode', type=str, required=False, default='train', choices=['train', 'finetune', 'resume'])
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
    parser.add_argument("--lr", type=float, required=False, default=3e-3)
    parser.add_argument("--decay", type=float, required=False, default=1e-2)
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

    if cfg.mode == 'finetune':
        assert cfg.ckpt != ''
    cfg.use_amp = not cfg.no_amp

    # s3dis
    cfg.num_classes = 13
    cfg.ignore_index = None

    prepare_exp(cfg)
    main(cfg)
