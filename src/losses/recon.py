import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class ReconLoss(nn.Module):
    """loss = alpha * L1 + (1 - alpha) * (1 - SSIM). Computed in float32 (safe under AMP).
    forward returns (loss, l1, ssim) with l1/ssim detached for logging."""

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        pred, target = pred.float(), target.float()
        l1 = F.l1_loss(pred, target)
        s = ssim(pred, target, data_range=1.0, size_average=True)
        return self.alpha * l1 + (1 - self.alpha) * (1 - s), l1.detach(), s.detach()
