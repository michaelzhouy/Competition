# -*- coding: utf-8 -*-
# @Time    : 2021/9/23 9:25 上午
# @Author  : Michael Zhouy
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
# from torch.utils.data.dataloader import default_collate
from PIL import Image


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


def data_loader(train_df, valid_df, test_df, batch_size=56, input_size=224, num_workers=16):
    train_loader = torch.utils.data.DataLoader(
        selfDataset(
            train_df['path'].values,
            train_df['label'].values,
            transforms.Compose([
                transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.CenterCrop(input_size),
                    # transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomVerticalFlip(),
                    # transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
            ])
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        # collate_fn=my_collate_fn
    )

    valid_loader = torch.utils.data.DataLoader(
        selfDataset(
            valid_df['path'].values,
            valid_df['label'].values,
            transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        # collate_fn=my_collate_fn
    )

    test_loader = torch.utils.data.DataLoader(
        selfDataset(
            test_df['path'].values,
            test_df['label'].values,
            transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        # collate_fn=my_collate_fn
    )
    return train_loader, valid_loader, test_loader
