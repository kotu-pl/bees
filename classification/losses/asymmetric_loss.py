import torch, torch.nn as nn

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=2.0, gamma_pos=0.0, clip=0.05, eps=1e-8, reduction='mean'):
        super().__init__()
        self.gn, self.gp = gamma_neg, gamma_pos
        self.clip, self.eps, self.red = clip, eps, reduction

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        if self.clip and self.clip > 0:
            p = p.clamp(self.clip, 1 - self.clip)

        pt_pos, pt_neg = p, 1.0 - p
        # log-likelihood
        loss = targets * torch.log(pt_pos + self.eps) + (1 - targets) * torch.log(pt_neg + self.eps)
        # asym. focusing
        pt = targets * pt_pos + (1 - targets) * pt_neg
        gamma = self.gp * targets + self.gn * (1 - targets)
        loss = -(torch.pow(1 - pt, gamma) * loss)

        if self.red == 'mean':  return loss.mean()
        if self.red == 'sum':   return loss.sum()
        return loss
