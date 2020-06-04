# -*- coding:utf-8 -*-
# Time   : 2020/5/10 18:50
# Email  : 15602409303@163.com
# Author : Zhou Yang

import numpy as np
import pandas as pd


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
