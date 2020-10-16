# -*- coding: utf-8 -*-
# @Time     : 2020/10/15 15:35
# @Author   : Michael_Zhouy

import pandas as pd
import datetime


def overfit_reducer(df, threshold=99.9):
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
        if zeros / len(df) * 100 > threshold:
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


def timestamp2string(timeStamp):
    """
    将时间戳转换为str对象
    @param timeStamp:
    @return:
    """
    try:
        d = datetime.datetime.fromtimestamp(timeStamp)
        str1 = d.strftime("%Y-%m-%d %H:%M:%S.%f")
        # 2015-08-28 16:43:37.283000'
        return str1
    except Exception as e:
        print(e)
        return ''


# 根据时间划分训练集、验证集和测试集
train = df.loc[df['observe_date'] < '2019-11-04', :]
valid = df.loc[(df['observe_date'] >= '2019-11-04') & (df['observe_date'] <= '2019-12-04'), :]
test = df.loc[df['observe_date'] > '2020-01-04', :]
