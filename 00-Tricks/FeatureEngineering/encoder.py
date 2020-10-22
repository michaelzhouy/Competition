# -*- coding: utf-8 -*-
# @Time     : 2020/10/15 14:59
# @Author   : Michael_Zhouy

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import gc
from sklearn.preprocessing import LabelEncoder





def low_freq_encode(df, cat_cols, freq=2):
    """
    将类别特征取值较少的归为一类
    @param df:
    @param cat_cols:
    @param freq: 取值频次
    @return:
    """
    for i in cat_cols:
        name_dict = dict(zip(*np.unique(df['name'], return_counts=True)))
        df['{}_low_freq'.format(i)] = df[i].apply(lambda x: -999 if name_dict[x] < freq else x)


def count_encode(df, cols=[]):
    """
    count编码
    @param df:
    @param cols:
    @return:
    """
    for col in cols:
        print(col)
        vc = df[col].value_counts(dropna=True, normalize=True)
        df[col + '_count'] = df[col].map(vc).astype('float32')


def label_encode(df, cols, verbose=True):
    """
    label encode
    @param df:
    @param cols:
    @param verbose:
    @return:
    """
    for col in cols:
        df[col], _ = df[col].factorize(sort=True)
        if df[col].max() > 32000:
            df[col] = df[col].astype('int32')
        else:
            df[col] = df[col].astype('int16')
        if verbose:
            print(col)


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


def cross_cat_num(df, cat_cols, num_cols):
    """
    类别特征与数据特征groupby统计
    @param df:
    @param cat_col: 类别特征
    @param num_col: 数值特征
    @return:
    """
    def max_min(s):
        return s.max() - s.min()

    def quantile(s, q=0.25):
        return s.quantile(q)

    for f1 in tqdm(cat_cols):
        g = df.groupby(f1, as_index=False)
        for f2 in tqdm(num_cols):
            tmp = g[f2].agg({
                '{}_{}_count'.format(f1, f2): 'count',
                '{}_{}_max'.format(f1, f2): 'max',
                '{}_{}_min'.format(f1, f2): 'min',
                '{}_{}_median'.format(f1, f2): 'median',
                '{}_{}_mean'.format(f1, f2): 'mean',
                '{}_{}_sum'.format(f1, f2): 'sum',
                '{}_{}_skew'.format(f1, f2): 'skew',
                '{}_{}_std'.format(f1, f2): 'std',
                '{}_{}_nunique'.format(f1, f2): 'nunique',
                '{}_{}_max_min'.format(f1, f2): max_min,
                '{}_{}_quantile_25'.format(f1, f2): lambda x: quantile(x, 0.25),
                '{}_{}_quantile_75'.format(f1, f2): lambda x: quantile(x, 0.75)
            })
            df = df.merge(tmp, on=f1, how='left')
            del tmp
            gc.collect()
    return df


def arithmetic(df, cross_features):
    """
    数值特征之间的加减乘除
    @param df:
    @param cross_features: 交叉用的数值特征
    @return:
    """
    for i in tqdm(range(len(cross_features))):
        for j in range(i + 1, len(cross_features)):
            colname_add = '{}_{}_add'.format(cross_features[i], cross_features[j])
            colname_substract = '{}_{}_subtract'.format(cross_features[i], cross_features[j])
            colname_multiply = '{}_{}c_multiply'.format(cross_features[i], cross_features[j])
            df[colname_add] = df[cross_features[i]] + df[cross_features[j]]
            df[colname_substract] = df[cross_features[i]] - df[cross_features[j]]
            df[colname_multiply] = df[cross_features[i]] * df[cross_features[j]]

    for f1 in tqdm(cross_features):
        for f2 in cross_features:
            if f1 != f2:
                colname_ratio = '{}_{}_ratio'.format(f1, f2)
                df[colname_ratio] = df[f1].values / (df[f2].values + 0.001)
    return df


def discretization(df, num_cols):
    """
    数值特征离散化
    @param df:
    @param num_cols:
    @return:
    """
    for f in num_cols:
        for bin in [20, 50, 100, 200]:
            df['{}_{}_bin'.format(f, bin)] = pd.cut(df[f], bin, duplicates='drop').apply(lambda x: x.left).astype(int)
