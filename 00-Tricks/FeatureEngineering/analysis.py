# -*- coding: utf-8 -*-
# @Time     : 2020/11/6 17:22
# @Author   : Michael_Zhouy
import pandas as pd
import matplotlib.pyplot as plt


def null_analysis(df, p='f24'):
    """
    特征的缺失情况随着时间推移的变化
    @param df:
    @param p:
    @return:
    """
    a = pd.DataFrame(df.groupby('ndays')[p].apply(lambda x: sum(pd.isnull(x))) / df.groupby('ndays')['ndays'].count()).reset_index()
    a.columns = ['ndays', p]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(a['ndays'], a[p])
    plt.axvline(61, color='r')
    plt.axvline(122, color='r')
    plt.axvline(153, color='r')
    plt.xlabel('ndays')
    plt.ylabel('miss_rate_' + p)
    plt.title('miss_rate_' + p)


def value_analysis(df, p='f24'):
    """
    特征取值随着时间推移的变化
    @param df:
    @param p:
    @return:
    """
    a = pd.DataFrame(df.groupby('ndays')[p].mean()).reset_index()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(a['ndays'], a[p])
    plt.axvline(61, color='r')
    plt.axvline(122, color='r')
    plt.axvline(153, color='r')
    plt.xlabel('ndays')
    plt.ylabel('mean_of_' + p)
    plt.title('distribution of ' + p)
