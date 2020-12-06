# -*- coding: utf-8 -*-
# @Time     : 2020/5/6 19:01
# @Author   : Michael_Zhouy

import math
from sklearn.utils.multiclass import type_of_target
import numpy as np
import pandas as pd


def woe(data, label, event=1):
    """

    Parameters
    ----------
    data
    label
    event

    Returns
    -------

    """
    negative_num = sum(label)
    positive_num = len(label) - negative_num

    check_target_binary(label)
    X1 = feature_discretion(data)

    columns = list(data.columns)
    res_woe = []
    res_iv = []
    for i in range(0, X1.shape[-1]):
        x = X1[:, i]
        woe_dict, iv1, df = woe_single(x, label, columns[i], event)

        refer = pd.DataFrame()
        refer[columns[i]] = x
        refer['target'] = label
        refer['neg_cnt'] = refer['target'].groupby(refer[columns[i]]).transform(lambda t: t.sum())
        refer['pos_cnt'] = refer['target'].groupby(refer[columns[i]]).transform(lambda t: len(t) - t.sum())
        refer = refer.drop_duplicates(columns[i], keep='first').reset_index(drop=True)
        refer = refer.sort_values(axis=0, by=columns[i], ascending=True)
        df['grouped_neg_cnt'] = refer['neg_cnt'].values
        df['grouped_pos_cnt'] = refer['pos_cnt'].values
        df['grouped_neg_cnt/(grouped_neg_cnt+grouped_pos_cnt)'] = df['grouped_neg_cnt'].values / (df['grouped_pos_cnt'].values + df['grouped_neg_cnt'].values)
        df['grouped_neg_cnt/total_grouped_neg_cnt'] = df['grouped_neg_cnt'].values / negative_num
        df['grouped_pos_cnt/total_grouped_pos_cnt'] = df['grouped_pos_cnt'].values / positive_num
        # df.to_excel(path + columns[i] + '.xlsx', index=False, encoding='gb18030')
        res_woe.append(woe_dict)
        res_iv.append(iv1)

    dict_IV = {'columns': data.columns, 'iv': res_iv}
    feature_IV = pd.DataFrame(dict_IV)
    feature_IV = feature_IV.sort_values(axis=0, by='iv', ascending=True)
    # feature_IV.to_excel(path + file + '.xlsx', index=False, encoding='gb18030')
    print(feature_IV)


def woe_single(x, y, x_name, event=1):
    """

    Parameters
    ----------
    x
    y
    x_name
    event

    Returns
    -------

    """
    check_target_binary(y)

    event_total, non_event_total = count_binary(y, event=event)
    x_labels = np.unique(x)
    woe_dict = {}
    woe_list = []
    iv_list = []
    iv = 0
    for x1 in x_labels:
        y1 = y[np.where(x == x1)[0]]
        event_count, non_event_count = count_binary(y1, event=event)
        rate_event = 1.0 * (event_count + 0.01) / (event_total + 0.01)
        rate_non_event = 1.0 * (non_event_count + 0.01) / (non_event_total + 0.01)
        woe1 = math.log(rate_event / rate_non_event)
        woe_dict[x1] = woe1
        iv1 = (rate_event - rate_non_event) * woe1
        woe_list.append(woe1)
        iv_list.append(iv1)
        iv += iv1
        # dict_IV = {'特征子分组': iv_dict.keys(), 'iv': iv_dict.values()}
    df = pd.DataFrame()
    df[x_name] = x_labels
    df['iv'] = iv_list
    df['woe'] = woe_list
    df = df.sort_values(axis=0, by=x_name, ascending=True)
    return woe_dict, iv, df


def count_binary(a, event=1):
    """

    Parameters
    ----------
    a
    event

    Returns
    -------

    """
    event_count = (a == event).sum()
    non_event_count = a.shape[-1] - event_count
    return event_count, non_event_count


def check_target_binary(y):
    """

    Parameters
    ----------
    y

    Returns
    -------

    """
    y_type = type_of_target(y)
    if y_type not in ['binary']:
        raise ValueError('Label type must be binary')


def feature_discretion(X):
    """

    Parameters
    ----------
    X

    Returns
    -------

    """
    temp = []
    for i in range(0, X.shape[-1]):
        columns = list(X.columns)
        print('-' * 20)
        print('Type: ', columns[i])
        x = X.iloc[:, i].values
        x_type = type_of_target(x)
        print('Type: ', x_type)
        if x_type == 'continuous':
            x1 = discrete(x)
            temp.append(x1)
        else:
            temp.append(x)
    return np.array(temp).T


def discrete(x, eps=0.000000001):
    """

    Parameters
    ----------
    x
    eps

    Returns
    -------

    """
    res = np.array([0] * x.shape[-1], dtype=int)
    max_ = np.max(x)
    min_ = np.min(x)
    n = 5
    for i in range(n):
        point1 = min_ + i * (max_ - min_) / n
        point2 = min_ + i * (max_ - min_) / n
        x1 = x[np.where((x > point1 - eps) & (x < point2 - eps))]
        mask = np.in1d(x, x1)
        res[mask] = (i + 1)
    return res
