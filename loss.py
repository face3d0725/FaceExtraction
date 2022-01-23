#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class OhemBCELoss(nn.Module):
    def __init__(self, thresh, n_min):
        super(OhemBCELoss, self).__init__()
        self.n_min = n_min
        self.criteria = nn.BCEWithLogitsLoss(reduction='none')
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()

    def forward(self, logits, labels):
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]

        else:
            loss = loss[: self.n_min]

        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1. - scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.Sigmoid()

    def forward(self, pr, gt, eps=1e-7):
        pr = self.activation(pr)
        tp = torch.sum(gt * pr)
        fp = torch.sum(pr) - tp
        fn = torch.sum(gt) - tp
        score = (2 * tp + eps) / (2 * tp + fn + fp + eps)
        return 1-score


class IoU(nn.Module):
    __name__ = 'iou_score'

    def __init__(self, threshold=0.0, eps=1e-7):
        super().__init__()
        self.threshold = threshold
        self.eps = 1e-7

    def iou(self, pr, gt):
        pr = (pr > self.threshold).type(pr.dtype)
        intersection = torch.sum(gt * pr) + self.eps
        union = torch.sum(gt) + torch.sum(pr) - intersection + self.eps
        return intersection / union

    def forward(self, pr, gt):
        return self.iou(pr, gt)

class Precision(nn.Module):
    __name__ = 'Precision_score'

    def __init__(self, threshold=0.0, eps=1e-7):
        super().__init__()
        self.threshold = threshold
        self.eps = 1e-7

    def iou(self, pr, gt):
        pr = (pr > self.threshold).type(pr.dtype)
        intersection = torch.sum(gt * pr) + self.eps
        union = torch.sum(pr)+ self.eps
        return intersection / union

    def forward(self, pr, gt):
        return self.iou(pr, gt)

if __name__ == '__main__':
    torch.manual_seed(15)
    criteria1 = OhemCELoss(thresh=0.7, n_min=16 * 20 * 20 // 16).cuda()
    criteria2 = OhemCELoss(thresh=0.7, n_min=16 * 20 * 20 // 16).cuda()
    net1 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net1.cuda()
    net1.train()
    net2 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net2.cuda()
    net2.train()

    with torch.no_grad():
        inten = torch.randn(16, 3, 20, 20).cuda()
        lbs = torch.randint(0, 19, [16, 20, 20]).cuda()
        lbs[1, :, :] = 255

    logits1 = net1(inten)
    logits1 = F.interpolate(logits1, inten.size()[2:], mode='bilinear')
    logits2 = net2(inten)
    logits2 = F.interpolate(logits2, inten.size()[2:], mode='bilinear')

    loss1 = criteria1(logits1, lbs)
    loss2 = criteria2(logits2, lbs)
    loss = loss1 + loss2
    print(loss.detach().cpu())
    loss.backward()
