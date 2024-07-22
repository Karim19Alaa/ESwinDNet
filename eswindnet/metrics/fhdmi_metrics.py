from skimage.metrics import peak_signal_noise_ratio as ski_psnr
from skimage.metrics import structural_similarity as ski_ssim
import numpy as np
import math
from torchvision.utils import make_grid
import torch


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """From https://github.com/CVMI-Lab/UHDM/blob/main/utils/common.py"""
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])

    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), padding=0, normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))

    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    
    
    return img_np.astype(out_type)

class PSNR(torch.nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()

    def __call__(self, img1, img2):
        return ski_psnr(tensor2img(img1), tensor2img(img2))
    
class SSIM(torch.nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()

    def __call__(self, img1, img2):
        return ski_ssim(tensor2img(img1), tensor2img(img2), multichannel=True, channel_axis=-1)
    
