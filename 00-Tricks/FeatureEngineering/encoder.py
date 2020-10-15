# -*- coding: utf-8 -*-
# @Time     : 2020/10/15 14:59
# @Author   : Michael_Zhouy

import numpy as np
from tqdm import tqdm
import gc


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


def cross_cat_num(df, cat_col, num_col):
    """
    类别特征与数据特征groupby统计
    @param df:
    @param cat_col: 类别特征
    @param num_col: 数值特征
    @return:
    """
    for f1 in tqdm(cat_col):
        g = df.groupby(f1, as_index=False)
        for f2 in tqdm(num_col):
            df_new = g[f2].agg({
                '{}_{}_max'.format(f1, f2): 'max',
                '{}_{}_min'.format(f1, f2): 'min',
                '{}_{}_median'.format(f1, f2): 'median',
                '{}_{}_mean'.format(f1, f2): 'mean',
                '{}_{}_sum'.format(f1, f2): 'sum',
                '{}_{}_skew'.format(f1, f2): 'skew',
                '{}_{}_nunique'.format(f1, f2): 'nunique'
            })
            df = df.merge(df_new, on=f1, how='left')
            del df_new
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
