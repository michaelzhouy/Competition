# -*- coding: utf-8 -*-
# @Time    : 2021/9/23 9:46 上午
# @Author  : Michael Zhouy
import pandas as pd
import os
from tqdm import tqdm
import torch
from glob import glob
from PIL import ImageFile
from util.prepare_data import load_data, split_data
from util.dataset import data_loader
os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'
ImageFile.LOAD_TRUNCATED_IMAGES = True


label_dic = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F'
}


def inference(net, test_loader, test):
    pred = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            net.eval()
            test_images, test_labels = data

            test_images = test_images.to('cuda')

            output = net(test_images)
            _, predicted = torch.max(output.data, 1)
            pred += predicted.tolist()

    test['label'] = pred

    test_tmp = pd.DataFrame({'path': glob('../data/valid_tmp/*.jpg')})
    test_tmp['id'] = test_tmp['path'].map(lambda x: x.split('/')[-1])
    test_tmp['label'] = -1
    test_tmp.loc[test_tmp['id'] == '7c7443.jpg', 'label'] = 4
    test_tmp.loc[test_tmp['id'] == '20a1af.jpg', 'label'] = 2
    test_tmp.loc[test_tmp['id'] == 'ace2f3.jpg', 'label'] = 4
    test_tmp.loc[test_tmp['id'] == 'd25f6d.jpg', 'label'] = 1
    test_tmp.loc[test_tmp['id'] == 'ff01e5.jpg', 'label'] = 3

    sub = pd.concat([test, test_tmp], ignore_index=True)
    sub['label'] = sub['label'].map(label_dic)
    print('sub shape: ', sub.shape)
    print('label null count: ', sub['label'].isnull().sum())

    sub[['id', 'label']].to_csv('../sub/sub4.csv', index=False)

data_dir = '../data/'
path = 'model/model_pth/efficientnet-b7/net_100.pth'
batch_size = 25
net = torch.load(path)

train, test = load_data(data_dir=data_dir)
train_df, valid_df = split_data(train, test_size=2000)

train_loader, val_loader, test_loader = data_loader(train_df, valid_df, test, batch_size=batch_size, input_size=224, num_workers=16)
inference(net, test_loader, test)
