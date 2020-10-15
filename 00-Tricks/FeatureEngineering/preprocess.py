# -*- coding: utf-8 -*-
# @Time     : 2020/10/15 15:35
# @Author   : Michael_Zhouy

import pandas as pd


def overfit_reducer(df, threshold):
    """
    计算每列中取值的分布，返回单一值占比达到阈值的列名
    @param df:
    @param threshold:
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
    计算每列的缺失率
    @param df:
    @return:
    """
    total = df.isnull().sum().sort_values(ascending=False)[df.isnull().sum().sort_values(ascending=False) != 0]
    percent = round(df.isnull().sum().sort_values(ascending=False) / len(df) * 100, 2)[round(df.isnull().sum().sort_values(ascending=False) / len(df) * 100, 2) != 0]
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
