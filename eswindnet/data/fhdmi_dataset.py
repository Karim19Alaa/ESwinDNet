import os
from torch.utils import data as data
from torchvision.io import read_image
from torchvision.transforms.v2.functional import crop
from torchvision.transforms.v2 import RandomCrop

from data.transforms import augment


class FHDMi(data.Dataset):

    def __init__(self, dataroot, phase, gt_size=None, use_hflip=True, use_vflip=True, use_rot=True, take=None):
        self.target_path = os.path.join(dataroot, 'target/target')
        self.target_list = sorted([os.path.join(self.target_path, file)
                                    for file in os.listdir(self.target_path) if file.endswith('.png')])
        
        self.source_path = os.path.join(dataroot, 'source/source')
        self.source_list = sorted([os.path.join(self.source_path, file)
                                    for file in os.listdir(self.source_path) if file.endswith('.png')])

        assert len(self.target_list) == len(self.source_list), 'Number of source and target images must be the same'

        if take is not None:
            self.target_list = self.target_list[:take]
            self.source_list = self.source_list[:take]

        self.phase = phase

        if self.phase == 'train':
            self.gt_size = gt_size
            self.use_hflip = use_hflip
            self.use_vflip = use_vflip
            self.use_rot = use_rot
            self.random_crop = RandomCrop(gt_size)

    def __getitem__(self, index):
        gt_path = self.target_list[index]
        img_gt = read_image(gt_path).float() / 255
        
        
        lq_path = self.source_list[index]
        img_lq = read_image(lq_path).float() / 255
        
        
        if self.phase == 'train':
            # random crop
            i, j, h, w = self.random_crop.get_params(img_gt, output_size=(self.gt_size, self.gt_size))
            img_gt = crop(img_gt, i, j, h, w)
            img_lq = crop(img_lq, i, j, h, w)

            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.use_hflip, self.use_vflip, self.use_rot)
        

        
        
        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.target_list)
    
