# coding=utf8
from __future__ import division
import pickle
import os
import torch
import torch.utils.data as data
import PIL.Image as Image
from PIL import ImageStat


class dataset(data.Dataset):
    def __init__(
            self,
            imgroot,
            anno_pd,
            unswap=None,
            totensor=None,
            train=False):
        self.root_path = imgroot
        self.paths = anno_pd['ImageName'].tolist()
        self.unswap = unswap
        self.anno_pd = anno_pd
        self.totensor = totensor
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = os.path.join(self.root_path, self.paths[item])
        img = self.pil_loader(img_path)
        img_unswap = self.unswap(img)
        img_unswap = self.totensor(img_unswap)

        return img_unswap, item

    def pil_loader(self, imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


def collate_fn2(batch):
    imgs = []
    item = []
    for sample in batch:
        imgs.append(sample[0])
        item.append(sample[1])
    return torch.stack(imgs, 0), item
