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

from backbone import Backbone, SegPartHead
from shapenetpart.configs import model_configs
from shapenetpart.dataset import ShapeNetPartNormalTest, shapenetpart_collate_fn, get_ins_mious
from utils import EasyConfig, setup_logger_dist, set_random_seed, resume_state, Timer, AverageMeter, Metric, \
    cal_model_params
from utils.logger import format_dict, format_list


def prepare_exp(cfg):
    exp_root = 'exp-test'
    exp_id = len(glob(f'{exp_root}/{cfg.exp}-*'))
    cfg.exp_name = f'{cfg.exp}-{exp_id:03d}'
    cfg.exp_dir = f'{exp_root}/{cfg.exp_name}'
    cfg.log_path = f'{cfg.exp_dir}/test.log'

    os.makedirs(cfg.exp_dir, exist_ok=True)
    setup_logger_dist(cfg.log_path, 0, name=cfg.exp_name)


def batched_bincount(x, dim, max_value):
    target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target

def knn_point(k, query, support=None):
    """Get the distances and indices to a fixed number of neighbors
        Args:
            support ([tensor]): [B, N, C]
            query ([tensor]): [B, M, C]

        Returns:
            [int]: neighbor idx. [B, M, K]
    """
    if support is None:
        support = query
    dist = torch.cdist(query, support)
    k_dist = dist.topk(k=k, dim=-1, largest=False, sorted=True)
    return k_dist.values, k_dist.indices

def torch_grouping_operation(features, idx):
    r"""from torch points kernels
    Parameters
    ----------
    features : torch.Tensor
        (B, C, N) tensor of features to group
    idx : torch.Tensor
        (B, npoint, nsample) tensor containing the indicies of features to group with

    Returns
    -------
    torch.Tensor
        (B, C, npoint, nsample) tensor
    """
    all_idx = idx.reshape(idx.shape[0], -1)
    all_idx = all_idx.unsqueeze(1).repeat(1, features.shape[1], 1)
    grouped_features = features.gather(2, all_idx)
    return grouped_features.reshape(idx.shape[0], features.shape[1], idx.shape[1], idx.shape[2])


def part_seg_refinement(pred, pos, cls, cls2parts, n=10):
    pred_np = pred.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):  # sample_idx
        parts = cls2parts[cls[shape_idx]]
        counter_part = Counter(pred_np[shape_idx])
        if len(counter_part) > 1:
            for i in counter_part:
                if counter_part[i] < n or i not in parts:
                    less_idx = np.where(pred_np[shape_idx] == i)[0]
                    less_pos = pos[shape_idx][less_idx]
                    knn_idx = knn_point(n + 1, torch.unsqueeze(less_pos, axis=0),
                                        torch.unsqueeze(pos[shape_idx], axis=0))[1]
                    neighbor = torch_grouping_operation(pred[shape_idx:shape_idx + 1].unsqueeze(1), knn_idx)[0][0]
                    counts = batched_bincount(neighbor, 1, cls2parts[-1][-1] + 1)
                    counts[:, i] = 0
                    pred[shape_idx][less_idx] = counts.max(dim=1)[1]
    return pred


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

    resume_state(model, cfg.ckpt)
    model.eval()

    writer = SummaryWriter(log_dir=cfg.exp_dir)
    timer = Timer(dec=1)
    timer_meter = AverageMeter()
    pbar = tqdm(enumerate(test_loader), total=test_loader.__len__(), desc='Test')
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
        p = gs.gs_points.p
        f = gs.gs_points.f
        pred = 0
        for i in range(cfg.voting):
            if i > 1:
                scale = torch.rand((3,), device=p.device) * 0.4 + 0.8
                p = p * scale
                height = p[:, 2:] * 4
                height -= height.min(dim=0, keepdim=True)[0]
                norm = f[:, :3]
                norm = norm * (scale[[1, 2, 0]] * scale[[2, 0, 1]])
                norm = torch.nn.functional.normalize(norm, p=2, dim=-1, eps=1e-8)
                f = torch.cat([norm, height], dim=-1)
                gs.gs_points.__update_attr__('p', p)
                gs.gs_points.__update_attr__('f', f)
            with autocast():
                pred = pred + model(gs, shape)
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
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--num_workers', type=int, required=False, default=12)

    # for test
    parser.add_argument("--voting", type=int, required=False, default=1)
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

    # shapenetpart
    cfg.num_classes = 50
    cfg.shape_classes = 16
    cfg.ignore_index = 255

    prepare_exp(cfg)
    main(cfg)
