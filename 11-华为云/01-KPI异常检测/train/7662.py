# -*- coding:utf-8 -*-
# Time   : 2020/10/24 16:59
# Email  : 15602409303@163.com
# Author : Zhou Yang

import os
import numpy as np
import datetime
import pandas as pd
import gc
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from naie.datasets import get_data_reference
from naie.context import Context
import warnings
warnings.filterwarnings('ignore')


def timestamp2string(timeStamp):
    try:
        d = datetime.datetime.fromtimestamp(timeStamp)
        str1 = d.strftime("%Y-%m-%d %H:%M:%S.%f")
        # 2015-08-28 16:43:37.283000'
        return str1
    except Exception as e:
        print(e)
        return ''


train = get_data_reference(dataset='data', dataset_entity='train').to_pandas_dataframe()
test = get_data_reference(dataset='data', dataset_entity='test').to_pandas_dataframe()

train['start_time'] = train['start_time'].map(lambda x: timestamp2string(x))
train['end_time'] = train['end_time'].map(lambda x: timestamp2string(x))

test['start_time'] = test['start_time'].map(lambda x: timestamp2string(x))
test['end_time'] = test['end_time'].map(lambda x: timestamp2string(x))

train['start_time'] = pd.to_datetime(train['start_time'])
train['end_time'] = pd.to_datetime(train['end_time'])
test['start_time'] = pd.to_datetime(test['start_time'])
test['end_time'] = pd.to_datetime(test['end_time'])

data = pd.concat([train, test])

data['hour'] = data['start_time'].map(lambda x: int(str(x)[11: 13]))
data['day'] = data['start_time'].map(lambda x: int(str(x)[8: 10]))
data['month'] = data['start_time'].map(lambda x: int(str(x)[5: 7]))
data['weekday'] = data['start_time'].map(lambda x: x.weekday())

data['hour'] = data['hour'].astype('category')
data['day'] = data['day'].astype('category')
data['month'] = data['month'].astype('category')
data['weekday'] = data['weekday'].astype('category')

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

# day, hour
# tmp = data.groupby(['kpi_id', 'day', 'hour'], as_index=False)['value'].agg({
#     'id_day_hour_mean': 'mean',
#     'id_day_hour_min': 'min',
#     'id_day_hour_max': 'max',
#     'id_day_hour_sum': 'sum',
#     'id_day_hour_median': 'median',
#     'id_day_hour_skew': 'skew'
# })
# data = data.merge(tmp, on=['kpi_id', 'day', 'hour'], how='left')
# del tmp
# gc.collect()

# tmp = data.groupby(['day', 'hour'], as_index=False)['value'].agg({
#     'day_hour_mean': 'mean',
#     'day_hour_min': 'min',
#     'day_hour_max': 'max',
#     'day_hour_sum': 'sum',
#     'day_hour_median': 'median',
#     'day_hour_skew': 'skew'
# })
# data = data.merge(tmp, on=['day', 'hour'], how='left')
# del tmp
# gc.collect()


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

used_cols = [i for i in data.columns if i not in ['kpi_id', 'label', 'start_time', 'end_time']]
train = data.loc[data['label'].notnull(), :]
y = train['label'].astype(int)
test = data.loc[data['label'].isnull(), :]
sub = test[['start_time', 'end_time', 'kpi_id']]

train = train[used_cols]
test = test[used_cols]

X = train.copy()
X_test = test.copy()
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=2020)

print('y_train mean: ', y_train.mean())
print('y_valid mean: ', y_valid.mean())

train_dataset = lgb.Dataset(X_train, y_train)
valid_dataset = lgb.Dataset(X_valid, y_valid, reference=train_dataset)
all_dataset = lgb.Dataset(X, y, reference=train_dataset)

params = {'objective': 'binary',
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
          'seed': 2020}
valid_model = lgb.train(params,
                        train_dataset,
                        valid_sets=[train_dataset, valid_dataset],
                        early_stopping_rounds=200,
                        num_boost_round=100000,
                        verbose_eval=300)
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

train_model = lgb.train(params,
                        all_dataset,
                        num_boost_round=valid_model.best_iteration + 20)
y_test_pred = np.where(train_model.predict(X_test) > threshold, 1, 0)

print('Test mean label: ', np.mean(y_test_pred))
sub['label'] = y_test_pred

out_path = os.path.join(Context.get_output_path(), "result" + '.csv')
with open(out_path, 'w') as f:
    sub.to_csv(f, index=False)
