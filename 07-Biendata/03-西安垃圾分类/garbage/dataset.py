#!usr/bin/env python  
#-*- coding:utf-8 _*- 
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 2019-09-07 20:27
公众号：AI成长社
知乎：https://www.zhihu.com/people/qlmx-61/columns
"""
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import pandas as pd
import six
import sys
from PIL import Image
import numpy as np
from torch.utils.data.dataloader import default_collate


class Dataset(Dataset):
    def __init__(self, root=None, transform=None, target_transform=None, to=None):
        if '.txt' in root:
            self.env = list(open(root))
        else:
            self.env = root

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        self.len = len(self.env) - 1

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        img_path, label = self.env[index].strip().split(',')

        try:
            img = Image.open(img_path).convert('RGB')
        except:
            print(img_path)
            print('Corrupted image for %d' % index)
            return self[index + 1]

        if self.transform is not None:
            # if img.layers == 1:
            #     print(img_path)
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)
        return (img, int(label))

class TestDataset(Dataset):
    def __init__(self, root=None, transform=None, target_transform=None, to=None):
        if '.txt' in root:
            self.env = list(open(root))
        else:
            self.env = root

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        self.len = len(self.env) - 1

        self.transform = transform
        self.target_transform = target_transform
        self.to = to

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        img_path, label = self.env[index].strip().split(',')

        try:
            img = Image.open(img_path).convert('RGB')
        except:
            print(img_path)
            print('Corrupted image for %d' % index)
            return self[index + 1]

        if self.transform is not None:
            img = self.transform(img)


        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, int(label))


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w+w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h+h_padding))

        # img.show()
        # resize
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class selfDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = torch.from_numpy(np.array([self.img_label[index]]))
        return img, label

    def __len__(self):
        return len(self.img_path)


def my_collate_fn(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def data_loader(train_df, valid_df, test_df, batch_size=56, input_size=380, num_workers=16):
    train_loader = torch.utils.data.DataLoader(
        selfDataset(
            train_df['path'].values,
            train_df['label'].values,
            transforms.Compose([
                transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.CenterCrop(input_size),
                    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            ])
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=my_collate_fn
    )

    valid_loader = torch.utils.data.DataLoader(
        selfDataset(
            valid_df['path'].values,
            valid_df['label'].values,
            transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=my_collate_fn
    )

    test_loader = torch.utils.data.DataLoader(
        selfDataset(
            test_df['path'].values,
            test_df['label'].values,
            transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=my_collate_fn
    )
    return train_loader, valid_loader, test_loader