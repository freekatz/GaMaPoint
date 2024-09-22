from math import log10
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import logging


def PSNR(mse, peak=1.):
    return 10 * log10((peak ** 2) / mse)


class SegMetric:
    def __init__(self, values=0.):
        assert isinstance(values, dict)
        self.miou = values.miou
        self.oa = values.get('oa', None) 
        self.miou = values.miou
        self.miou = values.miou


    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Metric():
    def __init__(self, num_classes=13, device='cuda'):
        self.n = num_classes
        self.label = torch.arange(num_classes, dtype=torch.int64, device=device).unsqueeze(1)
        self.device = device
        self.reset()

    def reset(self):
        # pred == label == i for i in 0...num_classes
        self.intersection = torch.zeros((self.n,), dtype=torch.int64, device=self.device)
        # pred == i or label == i
        self.union = torch.zeros((self.n,), dtype=torch.int64, device=self.device)
        # label == i
        self.count = torch.zeros((self.n,), dtype=torch.int64, device=self.device)
        #
        self.acc = 0.
        self.macc = 0.
        self.iou = [0.] * self.n
        self.miou = 0.

    def update(self, pred, label):
        pred = pred.max(dim=1)[1].unsqueeze(0) == self.label
        label = label.unsqueeze(0) == self.label
        self.tmp_c = label.sum(dim=1)
        self.count += self.tmp_c  # label.sum(dim=1)
        self.intersection += (pred & label).sum(dim=1)
        self.union += (pred | label).sum(dim=1)

    def calc_macc(self):
        macc = self.intersection / self.count
        macc = macc.mean()
        return macc

    def calc(self, digits=4):
        acc = self.intersection.sum() / self.count.sum()
        self.acc = round(acc.item(), digits)
        macc = self.calc_macc()
        self.macc = round(macc.item(), digits)
        iou = self.intersection / self.union
        self.iou = [round(i.item(), digits) for i in iou]
        miou = iou.mean()
        self.miou = round(miou.item(), digits)
        return acc, macc, miou, iou

