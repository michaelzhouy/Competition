# -*- coding:utf-8 -*-
# Time   : 2020/5/10 18:50
# Email  : 15602409303@163.com
# Author : Zhou Yang

import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import os
from scipy import stats
from joblib import Parallel, delayed
from datetime import datetime
from mlxtend.feature_selection import SequentialFeatureSelector

# 过滤警告
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

# DataFrame显示所有列
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)

# 读取压缩文件
df = pd.read_csv('../input/file.txt.gz', compression='gzip', header=0, sep=',', quotechar='"')

# 分块读取数据
reader = pd.read_csv('../input/train.csv', iterator=True)
df = reader.get_chunk(10000)
df.head()

# 数据存储为h5格式
df.to_hdf('data.h5', 'df')
# 读取h5文件
pd.read_hdf('data.h5')


# 节省内存读文件
def reduce_mem_usage(df):
    """
    iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    @param df:
    @return:
    """
    start_mem = df.memory_usage().sum()
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
            df[col] = df[col].astype('str')

    end_mem = df.memory_usage().sum()
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


# 读取文件下所有的文件，并合并成一个文件
def read_data(path):
    data_list = []
    for f in os.listdir(path):
        print(f)
        df = pd.read_csv(path + os.sep + f)
        print(df.shape)
        data_list.append(df)
        del df
        gc.collect()

    res = pd.concat(data_list, ignore_index=True)
    return res


# 筛选object特征
df_object = df.select_dtypes(include=['object'])
df_numerical = df.select_dtypes(exclude=['object'])


def get_same_set(train_df, test_df):
    """
    test中出现，train中没有出现的取值编码
    @param train_df:
    @param test_df:
    @return:
    """
    train_diff_test = set(train_df) - set(test_df)
    same = set(train_df) - train_diff_test
    test_diff_train = set(test_df) - same
    dic_ = {}
    cnt = 0
    for val in same:
        dic_[val] = cnt
        cnt += 1
    for val in train_diff_test:
        dic_[val] = cnt
        cnt += 1
    for val in test_diff_train:
        dic_[val] = cnt
        cnt += 1
    return dic_


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


# label encode
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


# groupby
gb = df.groupby(['user_id', 'page_id'], ax_index=False).agg(
    {'ad_price': {'max_price': np.max, 'min_price': np.min}})
gb.columns = ['user_id', 'page_id', 'min_price', 'max_price']
df = pd.merge(df, gb, on=['user_id', 'page_id'], how='left')


# rolling时间窗口
df['TTI'].groupby(df['id_road']).rolling('60min', closed='left', min_periods=6).mean()


"""
通话时间点的偏好
"""

# hour通话次数最高
tmp = df_voc.groupby('phone_no_m')['voc_hour'].agg(voc_hour_mode=lambda x: stats.mode(x)[0][0],        # 频次最高的元素
                                                   voc_hour_mode_count=lambda x: stats.mode(x)[1][0],  # 频次最高的元素的频次
                                                   voc_hour_nunique='nunique')
phone_no_m = phone_no_m.merge(tmp, on='phone_no_m', how='left')

# day通话次数最高
tmp = df_voc.groupby('phone_no_m')['voc_day'].agg(voc_day_mode=lambda x: stats.mode(x)[0][0],
                                                  voc_day_mode_count=lambda x: stats.mode(x)[1][0],
                                                  voc_day_nunique='nunique')
phone_no_m = phone_no_m.merge(tmp, on='phone_no_m', how='left')

del tmp
gc.collect()

# 每天的通话次数
voc_day_cnt_res = df_voc.groupby(['phone_no_m', 'voc_day'])['phone_no_m'].count().unstack()
for i in df_voc['voc_day'].unique():
    phone_no_m['voc_day{}_count'.format(i)] = phone_no_m['phone_no_m'].map(voc_day_cnt_res[i])

# 每天的通话人数
voc_day_nunique_res = df_voc.groupby(['phone_no_m', 'voc_day'])['opposite_no_m'].nunique().unstack()
for i in df_voc['voc_day'].unique():
    phone_no_m['voc_day{}_nunique'.format(i)] = phone_no_m['phone_no_m'].map(voc_day_nunique_res[i])

# 每天的通话时长
voc_day_call_dur_res = df_voc.groupby(['phone_no_m', 'voc_day'])['call_dur'].sum().unstack()
for i in df_voc['voc_day'].unique():
    phone_no_m['voc_day{}_call_dur_sum'.format(i)] = phone_no_m['phone_no_m'].map(voc_day_call_dur_res[i])