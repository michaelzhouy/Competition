# -*- coding: utf-8 -*-
# @Time    : 2021/10/16 9:09 上午
# @Author  : Michael Zhouy
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data.dataloader import default_collate
from PIL import Image
import cv2
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2
from prepare_data import load_data, split_data


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


def getTrainTransform(input_size):
    return Compose([
        Resize(input_size, input_size),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(p=0.25),
        HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.25),
        RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        CoarseDropout(p=0.5),
        Cutout(p=0.5),
        # ToTensorV2(p=1.0)
    ], p=1.
    )


def getValidTransform(input_size):
    return Compose([
        Resize(input_size, input_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        # ToTensorV2(p=1.0)
    ], p=1.
    )


def data_loader(train_df, valid_df, test_df, batch_size=56, input_size=384, num_workers=16):
    train_loader = torch.utils.data.DataLoader(
        selfDataset(
            train_df['path'].values,
            train_df['label'].values,
            getTrainTransform(input_size)
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
            getValidTransform(input_size)
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
            getValidTransform(input_size)
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=my_collate_fn
    )
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    data_dir = '../../data/'
    input_size = 384
    batch_size = 2
    num_workers = 2
    train, test = load_data(data_dir=data_dir)
    train_df, valid_df = split_data(train, test_size=2000)

    train_loader, valid_loader, test_loader = data_loader(train_df, valid_df, test, batch_size=batch_size,
                                                          input_size=input_size, num_workers=num_workers)
    for step, (imgs, image_labels) in enumerate(train_loader):
        # [c,h,w]->[h,w,c]
        # img = imgs[0].permute(1, 2, 0)
        # print(img.shape, image_labels)
        # print(isinstance(image_labels, list))
        # cv2.imwrite('aug.jpg', img.numpy())
        print('1')
        break
