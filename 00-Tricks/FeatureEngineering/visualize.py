# -*- coding: utf-8 -*-
# @Time     : 2020/10/23 17:54
# @Author   : Michael_Zhouy

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def box_plot(df, columns, frows, fcols, figsize=(80, 60)):
    """
    箱型图
    @param df:
    @param columns:
    @param rows:
    @param cols:
    @param figsize:
    @return:
    """
    plt.figure(figsize=figsize)
    i = 0
    for f in columns:
        i += 1
        plt.subplot(frows, fcols, i)
        sns.boxplot(df[f], orient='v', width=0.5)
        plt.ylabel(f, fontsize=36)
    plt.show()


def dist_plot(df, columns, frows, fcols, figsize=(80, 60)):
    """
    直方图
    @param df:
    @param columns:
    @param rows:
    @param cols:
    @param figsize:
    @return:
    """
    plt.figure(figsize=figsize)
    i = 0
    for f in columns:
        i += 1
        plt.subplot(frows, fcols, i)
        sns.distplot(df[f], fit=stats.norm)

        i += 1
        plt.subplot(frows, fcols, i)
        stats.probplot(df[f], plot=plt)
    plt.tight_layout()
    plt.show()


def kde_plot(train, test, columns, frows, fcols, figsize=(80, 60)):
    plt.figure(figsize=figsize)
    i = 0
    for f in columns:
        i += 1
        ax = plt.subplot(frows, fcols, i)
        sns.kdeplot(train[f], color='Red', shade=True)
        sns.kdeplot(test[f], color='Blue', shade=True)
        ax.set_xlabel(f)
        ax.set_ylabel('Frequency')
        ax.legend(['Train', 'Test'])
        i += 1
    plt.show()


def reg_plot(df, columns, frows, fcols, y='target', figsize=(80, 60)):
    plt.figure(figsize=figsize)
    i = 1
    for f in columns:
        ax = plt.subplot(frows, fcols, i)
        sns.regplot(x=f, y=y, data=df, ax=ax, scatter_kws={'marker': '.', 's': 3, 'alpha': 0.3}, line_kws={'color': 'k'})
        plt.xlabel(f)
        plt.ylabel(y)

        i += 1
        ax = plt.subplot(frows, fcols, i)
        sns.distplot(df[f].dropna())
        plt.xlabel(f)
    plt.show()
