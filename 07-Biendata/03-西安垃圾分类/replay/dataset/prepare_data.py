# -*- coding: utf-8 -*-
# @Time    : 2021/9/23 9:40 上午
# @Author  : Michael Zhouy
import pandas as pd
from sklearn.model_selection import train_test_split
from glob import glob

label_id_dic = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5
}


def load_data(data_dir):
    train = pd.read_csv(data_dir + 'train.csv')
    train['file_name'] = train['file_name'].map(lambda x: x.split('.')[0])
    train['label'] = train['category'].map(label_id_dic)
    train_img = pd.DataFrame({'path': glob(data_dir + 'train/*.*')})
    train_img['file_name'] = train_img['path'].map(lambda x: x.split('/')[-1].split('.')[0])
    train = train.merge(train_img, on='file_name', how='left')

    test = pd.DataFrame({'path': glob(data_dir + 'validation/*.*')})
    test['id'] = test['path'].map(lambda x: x.split('/')[-1])
    test['label'] = -1
    print('train Null counts: ', train.isnull().sum())
    print('train shape: ', train.shape)
    print('test  shape: ', test.shape)
    return train, test


def split_data(data, test_size=0.25):
    train_df, valid_df = train_test_split(data, test_size=test_size, random_state=6, stratify=data['label'])
    return train_df, valid_df
