import torch
from torch import nn
from torch.nn import functional as F

from utils.wavelets import WaveletTransform

class WaveletLossv2(nn.Module):
    def __init__(self, scale=2, params_path='eswindnet/utils/wavelet_weights_c2.pkl'):
        super(WaveletLossv2, self).__init__()

        self.dwt = WaveletTransform(scale=scale, params_path=params_path)

    def forward(self, pred, target):
        return F.mse_loss(self.dwt(pred[0]), self.dwt(target))

if __name__ == "__main__":
    loss = WaveletLossv2(scale=2)
    pred = torch.rand(1, 3, 256, 256)
    target = torch.rand(1, 3, 256, 256)
    print(loss(pred, target))