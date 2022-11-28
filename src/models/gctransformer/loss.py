import torch
import torch.nn as nn


class KLDivLoss(nn.Module):
    def __init__(self, padding_idx):
        super(KLDivLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        target_onehot = torch.zeros_like(x)
        target_onehot.scatter_(1, target.unsqueeze(1), 1.0)

        if self.padding_idx >= 0:
            target_onehot[:, self.padding_idx] = 0

        mask = torch.nonzero(target.data == self.padding_idx)  # noqa

        if mask.dim() > 0:
            target_onehot.index_fill_(0, mask.squeeze(), 0.0)

        loss = self.criterion(x, target_onehot)
        return loss


class LabelSmoothing(nn.Module):
    """ deprecated """
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)  # noqa

        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)

        self.true_dist = true_dist
        loss = self.criterion(x, true_dist.clone().detach())
        return loss
