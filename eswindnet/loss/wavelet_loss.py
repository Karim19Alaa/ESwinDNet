import torch
from torch import nn
from torch.nn import functional as F

from utils.wavelets import WaveletTransform

_reduction_modes = ['none', 'mean', 'sum']

class WaveletLoss(nn.Module):
    def __init__(self, texture_weight, sr_weight, lr_weight, params_path='eswindnet/utils/wavelet_weights_c2.pkl', scale=2, reduction='mean'):
        super(WaveletLoss, self).__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.dwt = WaveletTransform(scale=scale, params_path=params_path)
        self.reduction = reduction
        self.texture_weight = texture_weight
        self.sr_weight = sr_weight
        self.lr_weight = lr_weight

    def texture_loss(self, x, y, alpha=1.2, margin=0):
        xi = torch.sum(x * x, dim=2)
        yi = torch.sum(y * y, dim=2)
        
        out = F.relu(yi.mul(alpha) - xi + margin)
  
        return torch.mean(out)

    def forward(self, x, y, nc=3, alpha=1.2, margin=0):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        # """

        # print(pred.shape)
        x = self.dwt(x)
        y = self.dwt(y)

        xi = x.contiguous().view(x.size(0), -1, nc, x.size(2), x.size(3))
        yi = y.contiguous().view(y.size(0), -1, nc, y.size(2), y.size(3))

        x_l = xi[:, :nc, :, :]
        y_l = yi[:, :nc, :, :]
        x_h = xi[:, nc:, :, :]
        y_h = yi[:, nc:, :, :]
        
  
        return self.texture_loss(x_h, y_h, alpha, margin).mul(self.texture_weight) \
                + F.mse_loss(x_l, y_l).mul(self.lr_weight) \
                + F.mse_loss(x_h, y_h).mul(self.sr_weight)


