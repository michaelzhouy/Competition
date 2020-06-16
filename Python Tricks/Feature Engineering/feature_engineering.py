# -*- coding:utf-8 -*-
# Time   : 2020/5/10 18:50
# Email  : 15602409303@163.com
# Author : Zhou Yang

import numpy as np
import pandas as pd
from tqdm import tqdm
import gc


def overfit_reducer(df):
    """
    计算每列中取值的分布，返回单一值占比达99.74%的列名
    @param df:
    @return:
    """
    overfit = []
    for i in df.columns:
        counts = df[i].value_counts()
        zeros = counts.iloc[0]
        if zeros / len(df) * 100 > 99.94:
            overfit.append(i)
    return overfit


def missing_percentage(df):
    """
    计算缺失值占比
    @param df:
    @return:
    """
    total = df.isnull().sum().sort_values(ascending=False)[df.isnull().sum().sort_values(ascending=False) != 0]
    percent = round(df.isnull().sum().sort_values(ascending=False) / len(df) * 100, 2)[round(df.isnull().sum().sort_values(ascending = False) / len(df) * 100,2) != 0]
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


# groupby
gb = df.groupby(['user_id', 'page_id'], ax_index=False).agg(
    {'ad_price': {'max_price': np.max, 'min_price': np.min}})
gb.columns = ['user_id', 'page_id', 'min_price', 'max_price']
df = pd.merge(df, gb, on=['user_id', 'page_id'], how='left')


# 筛选object特征
df_object = df.select_dtypes(include=['object'])
df_numerical = df.select_dtypes(exclude=['object'])

# 将类别较少的取值归为一类
name_freq = 2
name_dict = dict(zip(*np.unique(df['name'], return_counts=True)))
df['name'] = df['name'].apply(lambda x: -999 if name_dict[x] < name_freq else x)

# 分组排名
df.groupby('uid')['time'].rank('dense')

# 根据时间划分训练集、验证集和测试集
train = df.loc[df['observe_date'] < '2019-11-04', :]
valid = df.loc[(df['observe_date'] >= '2019-11-04') & (df['observe_date'] <= '2019-12-04'), :]
test = df.loc[df['observe_date'] > '2020-01-04', :]


# count编码
def count_encode(df, cols=[]):
    for col in cols:
        print(col)
        vc = df[col].value_counts(dropna=True, normalize=True)
        df[col + '_count'] = df[col].map(vc).astype('float32')


# LABEL ENCODE
def label_encode(df, cols, verbose=True):
    for col in cols:
        df[col], _ = df[col].factorize(sort=True)
        if df[col].max() > 32000:
            df[col] = df[col].astype('int32')
        else:
            df[col] = df[col].astype('int16')
        if verbose:
            print(col)


# 交叉特征
def cross_cat_num(df, cat_col, num_col):
    for f1 in tqdm(cat_col):
        g = df.groupby(f1, as_index=False)
        for f2 in tqdm(num_col):
            df_new = g[f2].agg({
                '{}_{}_max'.format(f1, f2): 'max',
                '{}_{}_min'.format(f1, f2): 'min',
                '{}_{}_median'.format(f1, f2): 'median',
                '{}_{}_mean'.format(f1, f2): 'mean',
                '{}_{}_skew'.format(f1, f2): 'skew',
                '{}_{}_nunique'.format(f1, f2): 'nunique'
            })
            df = df.merge(df_new, on=f1, how='left')
            del df_new
            gc.collect()
    return df
