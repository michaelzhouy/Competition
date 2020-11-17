# -*- coding: utf-8 -*-
# @Time     : 2020/11/6 17:07
# @Author   : Michael_Zhouy
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc


def arithmetic(df, num_cols):
    """
    数值特征之间的加减乘除，x * x / y，log(x) / y
    @param df:
    @param num_cols: 交叉用的数值特征
    @return:
    """
    for i in tqdm(range(len(num_cols))):
        for j in range(i + 1, len(num_cols)):
            colname_add = '{}_{}_add'.format(num_cols[i], num_cols[j])
            colname_substract = '{}_{}_subtract'.format(num_cols[i], num_cols[j])
            colname_multiply = '{}_{}c_multiply'.format(num_cols[i], num_cols[j])
            df[colname_add] = df[num_cols[i]] + df[num_cols[j]]
            df[colname_substract] = df[num_cols[i]] - df[num_cols[j]]
            df[colname_multiply] = df[num_cols[i]] * df[num_cols[j]]

    for f1 in tqdm(num_cols):
        for f2 in num_cols:
            if f1 != f2:
                colname_ratio = '{}_{}_ratio'.format(f1, f2)
                df[colname_ratio] = df[f1].values / (df[f2].values + 0.001)
    return df


def low_freq_encode(df, cat_cols, freq=2):
    """
    将类别特征取值较少的归为一类
    @param df:
    @param cat_cols:
    @param freq: 取值频次
    @return:
    """
    for i in cat_cols:
        name_dict = dict(zip(*np.unique(df[i], return_counts=True)))
        df['{}_low_freq'.format(i)] = df[i].apply(lambda x: -999 if name_dict[x] < freq else x)


def count_encode(df, cat_cols):
    """
    类别特征的频次编码
    @param df:
    @param cat_cols:
    @return:
    """
    for col in cat_cols:
        print(col)
        vc = df[col].value_counts(dropna=True, normalize=True)
        df[col + '_count'] = df[col].map(vc).astype('float32')
    return df


def label_encode(df, cat_cols, verbose=True):
    """
    label encode
    @param df:
    @param cat_cols:
    @param verbose:
    @return:
    """
    for col in cat_cols:
        df[col], _ = df[col].factorize(sort=True)
        if df[col].max() > 32000:
            df[col] = df[col].astype('int32')
        else:
            df[col] = df[col].astype('int16')
        if verbose:
            print(col)
    return df


def cat_cat_stats(df, id_col, cat_cols):
    """
    类别特征之间的groupby统计特征
    @param df:
    @param id_col:
    @param cat_cols:
    @return:
    """
    for f1 in cat_cols:
        for f2 in cat_cols:
            if f1 != f2:
                tmp = df.groupby([id_col, f1], as_index=False)[f2].agg({
                    '{}_{}_cnt'.format(f1, f2): 'count',
                    '{}_{}_nunique'.format(f1, f2): 'nunique',
                    '{}_{}_mode'.format(f1, f2): lambda x: x.value_counts().index[0],  # 众数
                    '{}_{}_mode_cnt'.format(f1, f2): lambda x: x.value_counts().values[0]  # 众数出现的次数
                })
                tmp['{}_{}_rate'.format(f1, f2)] = tmp['{}_{}_nunique'.format(f1, f2)] / tmp['{}_{}_cnt'.format(f1, f2)]
                df = df.merge(tmp, on=[id_col, f1], how='left')
                del tmp
                gc.collect()
    return df


def cat_num_stats(df, cat_cols, num_cols):
    """
    类别特征与数据特征groupby统计特征，简单版
    @param df:
    @param cat_cols: 类别特征
    @param num_cols: 数值特征
    @return:
    """
    for f1 in tqdm(cat_cols):
        g = df.groupby(f1, as_index=False)
        for f2 in tqdm(num_cols):
            tmp = g[f2].agg({
                '{}_{}_max'.format(f1, f2): 'max',
                '{}_{}_min'.format(f1, f2): 'min',
                '{}_{}_median'.format(f1, f2): 'median',
                '{}_{}_mean'.format(f1, f2): 'mean',
                '{}_{}_sum'.format(f1, f2): 'sum',
                '{}_{}_skew'.format(f1, f2): 'skew',
                '{}_{}_std'.format(f1, f2): 'std'
            })
            df = df.merge(tmp, on=f1, how='left')
            del tmp
            gc.collect()
    return df


def cat_num_stats(df, cat_cols, num_cols):
    """
    类别特征与数据特征groupby统计特征，复杂版
    @param df:
    @param cat_cols: 类别特征
    @param num_cols: 数值特征
    @return:
    """
    def max_min(x):
        return x.max() - x.min()

    def q10(x):
        return x.quantile(0.1)

    def q20(x):
        return x.quantile(0.2)

    def q30(x):
        return x.quantile(0.3)

    def q40(x):
        return x.quantile(0.4)

    def q60(x):
        return x.quantile(0.6)

    def q70(x):
        return x.quantile(0.7)

    def q80(x):
        return x.quantile(0.8)

    def q90(x):
        return x.quantile(0.9)

    for f1 in tqdm(cat_cols):
        g = df.groupby(f1, as_index=False)
        for f2 in tqdm(num_cols):
            tmp = g[f2].agg({
                '{}_{}_cnt'.format(f1, f2): 'count',
                '{}_{}_max'.format(f1, f2): 'max',
                '{}_{}_min'.format(f1, f2): 'min',
                '{}_{}_median'.format(f1, f2): 'median',
                '{}_{}_mode'.format(f1, f2): lambda x: np.mean(pd.Series.mode(x)),
                # '{}_{}_mode'.format(f1, f2): lambda x: stats.mode(x)[0][0],
                # '{}_{}_mode'.format(f1, f2): lambda x: x.value_counts().index[0],
                '{}_{}_mean'.format(f1, f2): 'mean',
                '{}_{}_sum'.format(f1, f2): 'sum',
                '{}_{}_skew'.format(f1, f2): 'skew',
                '{}_{}_std'.format(f1, f2): 'std',
                '{}_{}_nunique'.format(f1, f2): 'nunique',
                '{}_{}_max_min'.format(f1, f2): lambda x: max_min(x),
                '{}_{}_q_10'.format(f1, f2): lambda x: q10(x),
                '{}_{}_q_20'.format(f1, f2): lambda x: q20(x),
                '{}_{}_q_30'.format(f1, f2): lambda x: q30(x),
                '{}_{}_q_40'.format(f1, f2): lambda x: q40(x),
                '{}_{}_q_60'.format(f1, f2): lambda x: q60(x),
                '{}_{}_q_70'.format(f1, f2): lambda x: q70(x),
                '{}_{}_q_80'.format(f1, f2): lambda x: q80(x),
                '{}_{}_q_90'.format(f1, f2): lambda x: q90(x),

            })
            df = df.merge(tmp, on=f1, how='left')
            del tmp
            gc.collect()
    return df


def topN(df, group_col, cal_col, N):
    """
    最受欢迎的元素及其频次
    @param df:
    @param group_col:
    @param cal_col:
    @param N: 欢迎程度, 0, 1, 2
    @return:
    """
    tmp = df.groupby(group_col, as_index=False)[cal_col].agg({
        '{}_{}_top_{}'.format(group_col, cal_col, N): lambda x: x.value_counts().index[N],
        '{}_{}_top_{}_cnt'.format(group_col, cal_col, N): lambda x: x.value_counts().values[N],
    })
    df = df.merge(tmp, on=group_col, how='left')
    del tmp
    gc.collect()
    return df


def binning(df, num_cols):
    """
    数值特征离散化
    @param df:
    @param num_cols:
    @return:
    """
    cat_cols = []
    for f in num_cols:
        for bins in [20, 50, 100, 200]:
            cat_cols.append('cut_{}_{}_bins'.format(f, bins))
            df['cut_{}_{}_bins'.format(f, bins)] = pd.cut(df[f], bins, duplicates='drop').apply(lambda x: x.left).astype(int)
    return df, cat_cols


# 获取TOP频次的位置信息，这里选Top3
mode_df = df.groupby(['ID', 'lat', 'lon'], as_index=False)['time'].agg({'mode_cnt': 'count'})
mode_df['rank'] = mode_df.groupby('ID')['mode_cnt'].rank(method='first', ascending=False)
for i in range(1, 4):
    tmp_df = mode_df[mode_df['rank'] == i]
    del tmp_df['rank']
    tmp_df.columns = ['ID', 'rank{}_mode_lat'.format(i), 'rank{}_mode_lon'.format(i), 'rank{}_mode_cnt'.format(i)]
    group_df = group_df.merge(tmp_df, on='ID', how='left')


def pivot(df, index, columns, func):
    df['tmp'] = 1
    tmp = df.pivot_table(values='tmp', index=index, columns=columns, aggfunc=func).fillna(0)
    tmp.columns = ['{}_{}'.format(columns, f) for f in tmp.columns]
    tmp.reset_index(inplace=True)
    return tmp
