import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, num_classes: int, smoothing: 0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), num_classes),
                                  device=targets.device).fill_(smoothing / (num_classes - 1)).scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
            return targets

    def forward(self, x, targets):
        targets = LabelSmoothingCrossEntropyLoss._smooth_one_hot(targets, x.size(-1), self.smoothing)
        log_softmax = F.log_softmax(x, dim=-1)
        if self.weight is not None:
            log_softmax = log_softmax * self.weight.unsqueeze(0)
        loss = -(targets * log_softmax).sum(-1)
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

