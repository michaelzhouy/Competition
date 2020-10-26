# -*- coding:utf-8 -*-
# Time   : 2020/10/24 10:51
# Email  : 15602409303@163.com
# Author : Zhou Yang

from naie.datasets import get_data_reference
from naie.context import Context
import os
import numpy as np
import datetime
import pandas as pd
import gc
from tqdm import tqdm
from joblib import Parallel, delayed
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import time
from util.util import *
import warnings
warnings.filterwarnings('ignore')


def get_shift_feature(df_, start, end, col='value', group='kpi_id_mdh'):
    df = df_.copy()
    add_feat = []
    for i in range(start, end + 1):
        add_feat.append('shift_{}_{}_{}'.format(col, group, i))
        df['{}_{}'.format(group, i)] = df[group] + i
        df_last = df[df[col].notnull()].set_index('{}_{}'.format(col, i))
        df['shift_{}_{}_{}'.format(col, group, i)] = df[group].map(df_last[col])
        del df['{}_{}'.format(group, i)]
        del df_last
        gc.collect()
    return df, add_feat


def get_adjoin_feature(df_, start, end, space, col='value', group='kpi_id_mdh'):
    df = df_.copy()
    add_feat = []
    for i in range(start, end+1):
        add_feat.append('adjoin_{}_{}_{}_{}_{}_sum'.format(col, group, i, i + space, space)) # 求和
        add_feat.append('adjoin_{}_{}_{}_{}_{}_mean'.format(col, group, i, i + space, space)) # 均值
        add_feat.append('adjoin_{}_{}_{}_{}_{}_diff'.format(col, group, i, i + space, space)) # 首尾差值
        add_feat.append('adjoin_{}_{}_{}_{}_{}_ratio'.format(col, group, i, i + space, space)) # 首尾比例
        df['adjoin_{}_{}_{}_{}_{}_sum'.format(col, group, i, i + space, space)] = 0
        for j in range(0, space+1):
            df['adjoin_{}_{}_{}_{}_{}_sum'.format(col, group, i, i + space, space)] = (df['adjoin_{}_{}_{}_{}_{}_sum'.format(col, group, i, i + space, space)]
                                                                                       + df['shift_{}_{}_{}'.format(col, group, i + j)])
        df['adjoin_{}_{}_{}_{}_{}_mean'.format(col, group, i, i + space, space)] = (df['adjoin_{}_{}_{}_{}_{}_sum'.format(col, group, i, i + space, space)].values
                                                                                   / (space + 1))
        df['adjoin_{}_{}_{}_{}_{}_diff'.format(col, group, i, i + space, space)] = (df['shift_{}_{}_{}'.format(col, group, i)].values
                                                                                    - df['shift_{}_{}_{}'.format(col, group, i + space)])
        df['adjoin_{}_{}_{}_{}_{}_ratio'.format(col, group, i, i + space, space)] = (df['shift_{}_{}_{}'.format(col, group, i)].values
                                                                                     / df['shift_{}_{}_{}'.format(col, group, i + space)])
    return df, add_feat


def get_series_feature(df_, start, end, col='value', group='kpi_id_mdh', types=['sum', 'mean', 'min', 'max', 'std', 'ptp']):
    df = df_.copy()
    add_feat = []
    li = []
    df['series_{}_{}_{}_{}_sum'.format(col, group, start, end)] = 0
    for i in range(start, end + 1):
        li.append('shift_{}_{}_{}'.format(col, group, i))
    df['series_{}_{}_{}_{}_sum'.format(col, group, start, end)] = df[li].apply(np.sum, axis=1)
    df['series_{}_{}_{}_{}_mean'.format(col, group, start, end)] = df[li].apply(np.mean, axis=1)
    df['series_{}_{}_{}_{}_min'.format(col, group, start, end)] = df[li].apply(np.min, axis=1)
    df['series_{}_{}_{}_{}_max'.format(col, group, start, end)] = df[li].apply(np.max, axis=1)
    df['series_{}_{}_{}_{}_std'.format(col, group, start, end)] = df[li].apply(np.std, axis=1)
    df['series_{}_{}_{}_{}_ptp'.format(col, group, start, end)] = df[li].apply(np.ptp, axis=1)
    for typ in types:
        add_feat.append('series_{}_{}_{}_{}_{}'.format(col, group, start, end, typ))
    return df, add_feat

print(time.strftime('%Y%m%d'))
train = get_data_reference(dataset='data', dataset_entity='train').to_pandas_dataframe()
test = get_data_reference(dataset='data', dataset_entity='test').to_pandas_dataframe()

submission = test[['start_time', 'end_time', 'kpi_id']]

data = pd.concat([train, test])

data = data.groupby('kpi_id', as_index=False).apply(lambda x: x.sort_values('start_time'))

data['start_time_str'] = data['start_time'].map(lambda x: timestamp2string(x))

data = get_datetime(data, 'start_time_str', type='hour')
data = get_datetime(data, 'start_time_str', type='day')
data = get_datetime(data, 'start_time_str', type='month')
data = get_datetime(data, 'start_time_str', type='weekday')

data['mdh'] = ((data['month'] - 8) * 31 + data['day']) * 24 + data['hour']
data['kpi_id_num'] = data['kpi_id'].map(dict(zip(data['kpi_id'].unique(), range(data['kpi_id'].nunique()))))
data['kpi_id_mdh'] = data['kpi_id_num'] * 10000 + data['mdh']


for col in ['hour', 'day', 'month', 'weekday']:
    data[col] = data[col].astype('category')

# hour
tmp = data.groupby(['kpi_id', 'hour'], as_index=False)['value'].agg({
    'id_hour_mean': 'mean',
    'id_hour_min': 'min',
    'id_hour_max': 'max',
    'id_hour_sum': 'sum',
    'id_hour_median': 'median',
    'id_hour_skew': 'skew'
})
data = data.merge(tmp, on=['kpi_id', 'hour'], how='left')
del tmp
gc.collect()

tmp = data.groupby(['hour'], as_index=False)['value'].agg({
    'hour_mean': 'mean',
    'hour_min': 'min',
    'hour_max': 'max',
    'hour_sum': 'sum',
    'hour_median': 'median',
    'hour_skew': 'skew'
})
data = data.merge(tmp, on=['hour'], how='left')
del tmp
gc.collect()

# day
tmp = data.groupby(['kpi_id', 'day'], as_index=False)['value'].agg({
    'id_day_mean': 'mean',
    'id_day_min': 'min',
    'id_day_max': 'max',
    'id_day_sum': 'sum',
    'id_day_median': 'median',
    'id_day_skew': 'skew'
})
data = data.merge(tmp, on=['kpi_id', 'day'], how='left')
del tmp
gc.collect()

tmp = data.groupby(['day'], as_index=False)['value'].agg({
    'day_mean': 'mean',
    'day_min': 'min',
    'day_max': 'max',
    'day_sum': 'sum',
    'day_median': 'median',
    'day_skew': 'skew'
})
data = data.merge(tmp, on=['day'], how='left')
del tmp
gc.collect()

# month
tmp = data.groupby(['kpi_id', 'month'], as_index=False)['value'].agg({
    'id_month_mean': 'mean',
    'id_month_min': 'min',
    'id_month_max': 'max',
    'id_month_sum': 'sum',
    'id_month_median': 'median',
    'id_month_skew': 'skew'
})
data = data.merge(tmp, on=['kpi_id', 'month'], how='left')
del tmp
gc.collect()

tmp = data.groupby(['month'], as_index=False)['value'].agg({
    'month_mean': 'mean',
    'month_min': 'min',
    'month_max': 'max',
    'month_sum': 'sum',
    'month_median': 'median',
    'month_skew': 'skew'
})
data = data.merge(tmp, on=['month'], how='left')
del tmp
gc.collect()

# weekday
tmp = data.groupby(['kpi_id', 'weekday'], as_index=False)['value'].agg({
    'id_weekday_mean': 'mean',
    'id_weekday_min': 'min',
    'id_weekday_max': 'max',
    'id_weekday_sum': 'sum',
    'id_weekday_median': 'median',
    'id_weekday_skew': 'skew'
})
data = data.merge(tmp, on=['kpi_id', 'weekday'], how='left')
del tmp
gc.collect()

tmp = data.groupby(['weekday'], as_index=False)['value'].agg({
    'weekday_mean': 'mean',
    'weekday_min': 'min',
    'weekday_max': 'max',
    'weekday_sum': 'sum',
    'weekday_median': 'median',
    'weekday_skew': 'skew'
})
data = data.merge(tmp, on=['weekday'], how='left')
del tmp
gc.collect()

# weekday, hour
tmp = data.groupby(['kpi_id', 'weekday', 'hour'], as_index=False)['value'].agg({
    'id_weekday_hour_mean': 'mean',
    'id_weekday_hour_min': 'min',
    'id_weekday_hour_max': 'max',
    'id_weekday_hour_sum': 'sum',
    'id_weekday_hour_median': 'median',
    'id_weekday_hour_skew': 'skew'
})
data = data.merge(tmp, on=['kpi_id', 'weekday', 'hour'], how='left')
del tmp
gc.collect()

tmp = data.groupby(['weekday', 'hour'], as_index=False)['value'].agg({
    'weekday_hour_mean': 'mean',
    'weekday_hour_min': 'min',
    'weekday_hour_max': 'max',
    'weekday_hour_sum': 'sum',
    'weekday_hour_median': 'median',
    'weekday_hour_skew': 'skew'
})
data = data.merge(tmp, on=['weekday', 'hour'], how='left')
del tmp
gc.collect()


# 时序特征
stat_feat = []
# 平移
start, end = 1, 96
data, add_feat = get_shift_feature(data, start, end, col='value', group='kpi_id_mdh')

# 相邻
start, end = 1, 95
data, add_feat = get_adjoin_feature(data, start, end, space=1, col='value', group='kpi_id_mdh')
stat_feat = stat_feat + add_feat
start, end = 1, 94
data, add_feat = get_adjoin_feature(data, start, end, space=2, col='value', group='kpi_id_mdh')
stat_feat = stat_feat + add_feat
start, end = 1, 93
data, add_feat = get_adjoin_feature(data, start, end, space=3, col='value', group='kpi_id_mdh')
stat_feat = stat_feat + add_feat
start, end = 1, 92
data, add_feat = get_adjoin_feature(data, start, end, space=4, col='value', group='kpi_id_mdh')
stat_feat = stat_feat + add_feat
start, end = 1, 91
data, add_feat = get_adjoin_feature(data, start, end, space=5, col='value', group='kpi_id_mdh')
stat_feat = stat_feat + add_feat
start, end = 1, 90
data, add_feat = get_adjoin_feature(data, start, end, space=6, col='value', group='kpi_id_mdh')
stat_feat = stat_feat + add_feat

# 连续
start, end = 1, 3
data, add_feat = get_series_feature(data, start, end)
stat_feat = stat_feat + add_feat
start, end = 1, 5
data, add_feat = get_series_feature(data, start, end)
stat_feat = stat_feat + add_feat
start, end = 1, 7
data, add_feat = get_series_feature(data, start, end)
stat_feat = stat_feat + add_feat


train = data.loc[data['label'].notnull(), :]
y = train['label'].astype(int)
test = data.loc[data['label'].isnull(), :]
sub = test[['start_time', 'end_time', 'kpi_id']]
del data
gc.collect()

train.drop(['kpi_id', 'start_time', 'end_time'], axis=1, inplace=True)
test.drop(['kpi_id', 'start_time', 'end_time'], axis=1, inplace=True)

print('train:')
print(train[['start_time_str']].head())
print(train['start_time_str'].dtype)

train_train = train.loc[(train['start_time_str'] >= '2019-08-01 00:00:00') & (train['start_time_str'] <= '2019-09-15 23:00:00'), :]
train_valid = train.loc[(train['start_time_str'] >= '2019-09-16 00:00:00') & (train['start_time_str'] <= '2019-09-22 23:00:00'), :]

print('train_train.shape: \n', train_train.shape)
print('train_valid.shape: \n', train_valid.shape)
print('test.shape: \n', test.shape)

used_cols = [i for i in train_train.columns if i not in ['kpi_id', 'label', 'start_time', 'end_time', 'start_time_str']]

X_train = train_train[used_cols]
X_valid = train_valid[used_cols]
X_test = test[used_cols]
y_train = train_train['label']
y_valid = train_valid['label']

print('y_train mean: \n', y_train.mean())
print('y_valid mean: \n', y_valid.mean())

used_cols = X_train.columns.to_list()

train_dataset = lgb.Dataset(X_train, y_train)
valid_dataset = lgb.Dataset(X_valid, y_valid, reference=train_dataset)
all_dataset = lgb.Dataset(train[used_cols], y, reference=train_dataset)

params = {
    'objective': 'binary',
    'boosting': 'gbdt',
    'metric': 'auc',
    # 'metric': 'None',  # 用自定义评估函数是将metric设置为'None'
    'learning_rate': 0.1,
    'num_leaves': 31,
    'lambda_l1': 0,
    'lambda_l2': 1,
    'num_threads': 23,
    'min_data_in_leaf': 20,
    'first_metric_only': True,
    'is_unbalance': True,
    'max_depth': -1,
    'seed': 2020
}

valid_model = lgb.train(
    params,
    train_dataset,
    valid_sets=[valid_dataset, train_dataset],
    early_stopping_rounds=200,
    num_boost_round=1000000,
    verbose_eval=300
)

pred = valid_model.predict(X_valid)

f1_best = 0
for i in np.arange(0.1, 1, 0.01):
    y_valid_pred = np.where(pred > i, 1, 0)
    f1 = np.round(f1_score(y_valid, y_valid_pred), 5)
    if f1 > f1_best:
        threshold = i
        f1_best = f1

print('threshold: ', threshold)
y_valid_pred = np.where(pred > threshold, 1, 0)
print('Valid F1: ', np.round(f1_score(y_valid, y_valid_pred), 5))
print('Valid mean label: ', np.mean(y_valid_pred))

train_model = lgb.train(
    params,
    all_dataset,
    num_boost_round=valid_model.best_iteration + 20
)

y_test_pred = np.where(train_model.predict(X_test) > threshold, 1, 0)

print('Test mean label: ', np.mean(y_test_pred))
sub['label'] = y_test_pred
submission = submission.merge(sub, on=['start_time', 'end_time', 'kpi_id'], how='left')

out_path = os.path.join(Context.get_output_path(), "result" + '.csv')
with open(out_path, 'w') as f:
    submission.to_csv(f, index=False)
