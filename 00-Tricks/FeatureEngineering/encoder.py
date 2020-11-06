# -*- coding: utf-8 -*-
# @Time     : 2020/10/15 14:59
# @Author   : Michael_Zhouy

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import gc
from sklearn.preprocessing import LabelEncoder
from scipy import stats





def train_test_label_encode(df, cat_col, type='save', path='./'):
    """
    train和test分开label encode
    save的食用方法
    for i in cat_cols:
        train = train_test_label_encode(train, i, 'save', './')
        train[i] = train[i].astype('category')
    load的食用方法：
    for i in cat_cols:
        d = train_test_label_encode(test, i, 'load', '../train_code/')
        test[i] = test[i].map(d)
        test[i] = test[i].astype('category')
    @param df:
    @param cat_col:
    @param type: 'save' 'load'
    @param path:
    @return:
    """
    def save_obj(obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f)

    def load_obj(name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

    if type == 'save':
        print(cat_col)
        d = dict(zip(df[cat_col].unique(), range(df[cat_col].nunique())))
        df[cat_col] = df[cat_col].map(d)
        np.save(path + '{}.npy'.format(cat_col), d)
        return df
    elif type == 'load':
        d = np.load(path + '{}.npy'.format(cat_col), allow_pickle=True).item()
        return d


def train_test_label_encode_2(df, cat_col, type='save', path='./'):
    """
    train和test分开label encode
    save的食用方法
    for i in cat_cols:
        train = train_test_label_encode_2(train, i, 'save', './')
        train[i] = train[i].astype('category')
    load的食用方法：
    for i in cat_cols:
        d = train_test_label_encode_2(test, i, 'load', '../train_code/')
        test[i] = test[i].map(d)
        test[i] = test[i].astype('category')
    @param df:
    @param cat_col:
    @param type: 'save' 'load'
    @param path:
    @return:
    """
    if type == 'save':
        print(cat_col)
        le = LabelEncoder()
        le.fit(df[cat_col])
        le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
        df[cat_col] = df[cat_col].map(le_dict)
        np.save(path + '{}.npy'.format(cat_col), d)
        return df
    elif type == 'load':
        print(cat_col)
        d = np.load(path + '{}.npy'.format(cat_col), allow_pickle=True).item()
        return d









