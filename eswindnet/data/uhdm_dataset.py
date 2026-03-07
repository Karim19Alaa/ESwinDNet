import os
from torch.utils import data as data
from torchvision.io import read_image
from torchvision.transforms.v2.functional import crop
from torchvision.transforms.v2 import RandomCrop

from data.transforms import augment


def _list_image_files_recursively(data_dir):
    file_list = []
    for home, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith("gt.jpg"):
                file_list.append(os.path.join(home, filename))
    file_list.sort()
    return file_list


class UHDM(data.Dataset):

    def __init__(
        self,
        dataroot,
        phase,
        gt_size=None,
        use_hflip=True,
        use_vflip=True,
        use_rot=True,
        take=None,
    ):
        self.image_list = _list_image_files_recursively(dataroot)
        if take is not None:
            self.image_list = self.image_list[:take]

        self.phase = phase

        if self.phase == "train":
            self.gt_size = gt_size
            self.use_hflip = use_hflip
            self.use_vflip = use_vflip
            self.use_rot = use_rot
            if gt_size:
                self.random_crop = RandomCrop(gt_size)
            else:
                self.random_crop = None

    def __getitem__(self, index):
        gt_path = self.image_list[index]
        img_gt = read_image(gt_path).float() / 255

        lq_path = os.path.join(
            os.path.split(gt_path)[0], os.path.split(gt_path)[-1][0:4] + "_moire.jpg"
        )
        img_lq = read_image(lq_path).float() / 255

        if self.phase == "train":
            if self.random_crop:
                # random crop
                i, j, h, w = self.random_crop.get_params(
                    img_gt, output_size=(self.gt_size, self.gt_size)
                )
                img_gt = crop(img_gt, i, j, h, w)
                img_lq = crop(img_lq, i, j, h, w)

            # flip, rotation
            img_gt, img_lq = augment(
                [img_gt, img_lq], self.use_hflip, self.use_vflip, self.use_rot
            )

        return {"lq": img_lq, "gt": img_gt, "lq_path": lq_path, "gt_path": gt_path}

    def __len__(self):
        return len(self.image_list)
