from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import functional as F, InterpolationMode


class Resize:
    def __init__(self, size: int | tuple[int]):
        """
        缩放图像到 size 大小，对于二值图 label 采用最邻近插值。

        Args:
            size (int | tuple): 要缩放到的大小。
        """
        if isinstance(size, tuple):
            assert len(size) <= 2, "size 的长度不能超过 2。"

            if len(size) == 1:
                self.size = (size[0], size[0])
            else:
                self.size = size
        elif isinstance(size, int):
            self.size = (size, size)
        else:
            raise TypeError("请输入 (h, w) 或 int")

    def __call__(self, img_tuple):
        img, label = img_tuple
        img = F.resize(img, [self.size[0], self.size[1]])

        if label is not None:
            label = F.resize(label, [self.size[0], self.size[1]], interpolation=InterpolationMode.BILINEAR)

        if img.max() > 1:
            img = img / img.max()
            if label is not None:
                label = label / label.max()
        else:
            raise RuntimeError("图像过暗，无法识别。")

        return img, label


class RandomFlipCrop:
    def __init__(self, p: float, size: int):
        """
        对图像做随机上下反转和随机裁剪。

        Args:
            p (float): 随机上下反转的概率。
            size (int): 随机裁剪的大小。
        """
        assert isinstance(size, int), "size 必须是 int。"

        self.p = p
        self.size = size

    def __call__(self, img_tuple):
        img, label = img_tuple
        if label is not None:
            combined = torch.cat((img, label), dim=0)
            combined = transforms.RandomVerticalFlip(self.p)(combined)
            combined = transforms.RandomCrop(self.size)(combined)

            return combined[:img.shape[0], :, :], combined[img.shape[0]:, :, :]

        else:
            img = transforms.RandomVerticalFlip(self.p)(img)
            img = transforms.RandomCrop(self.size)(img)

            return img, label


class Normalize:
    def __call__(self, img_tuple):
        img, label = img_tuple
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        return img, label


class U2NETDataSet(Dataset):
    def __init__(self, img_lst: list, label_lst: list, transform: Optional[transforms.Compose] = None):
        self.img_lst = img_lst
        self.label_lst = label_lst
        self.transform = transform

    def __len__(self):
        return len(self.img_lst)

    def __getitem__(self, idx):
        img = read_image(self.img_lst[idx]).to(torch.float)

        if len(self.label_lst):
            label = read_image(self.label_lst[idx], ImageReadMode.GRAY).to(torch.float)
        else:
            label = None

        if self.transform:
            img, label = self.transform((img, label))

        return img, label
