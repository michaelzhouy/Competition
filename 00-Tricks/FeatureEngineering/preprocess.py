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
        # str类型
        str = d.strftime('%Y-%m-%d %H:%M:%S.%f')
        return str
    except Exception as e:
        print(e)
        return ''


def get_datetime(df, time_col, type='hour'):
    """
    获取年、月、日、小时等
    @param df:
    @param time_col:
    @param type: 'hour' 'day' 'month' 'year' 'weekday'
    @return:
    """
    df[time_col + '_datetime'] = pd.to_datetime(df[time_col])
    if type == 'hour':
        df['hour'] = df[time_col + '_datetime'].map(lambda x: int(str(x)[11: 13]))
    elif type == 'day':
        df['day'] = df[time_col + '_datetime'].map(lambda x: int(str(x)[8: 10]))
    elif type == 'month':
        df['month'] = df[time_col + '_datetime'].map(lambda x: int(str(x)[5: 7]))
    elif type == 'year':
        df['year'] = df[time_col + '_datetime'].map(lambda x: int(str(x)[0: 4]))
    elif type == 'weekday':
        df['weekday'] = df[time_col + '_datetime'].map(lambda x: x.weekday())
    del df[time_col + '_datetime']
    return df


def time_split(df):
    """
    根据时间划分训练集、验证集和测试集
    @param df:
    @return:
    """
    train = df.loc[df['observe_date'] < '2019-11-04', :]
    valid = df.loc[(df['observe_date'] >= '2019-11-04') & (df['observe_date'] <= '2019-12-04'), :]
    test = df.loc[df['observe_date'] > '2020-01-04', :]
    return train, valid, test
