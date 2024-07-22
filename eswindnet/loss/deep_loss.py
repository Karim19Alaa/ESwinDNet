import torch
from torch import nn as nn
from torch.nn import functional as F

# from basicsr.utils.registry import LOSS_REGISTRY

_reduction_modes = ['none', 'mean', 'sum']

class DeepLoss(nn.Module):
    def __init__(self, loss, reduction='mean'):
        super(DeepLoss, self).__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss = loss

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        # """
    
        gt2 = F.interpolate(target, scale_factor=0.5, mode='bilinear', align_corners=False)
        gt3 = F.interpolate(target, scale_factor=0.25, mode='bilinear', align_corners=False)

        return self.loss(pred[0], target) + self.loss(pred[1], gt2) + self.loss(pred[2], gt3)

