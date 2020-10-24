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
from util.util import *
import warnings
warnings.filterwarnings('ignore')


train = get_data_reference(dataset='data', dataset_entity='train').to_pandas_dataframe()
test = get_data_reference(dataset='data', dataset_entity='test').to_pandas_dataframe()

submission = test[['start_time', 'end_time', 'kpi_id']]

kpis_list = ["9415ac3c-cae9-4906-b65c-bc9c7a732c30", "600a5d6e-fd61-43a9-9857-783cec807879",
             "31997140-314b-459a-a69c-c9d3e31ec1a1", "a113b2a7-0a80-4ef6-8dac-b35ab1ca4f98",
             "e6999100-d229-41c1-9370-f7b5ff315b8b", "3fe4d11f-4e06-4725-bf16-19db40a7a3e1",
             "681cbb98-68e2-4d9a-af4f-9efde2768a5e", "29374201-b68d-4714-a2ee-4772ac52447f",
             "355eda04-426e-4c9f-aba0-6481db290068", "4f4936b1-1a23-4eba-9e69-41a304e9b1a1",
             "b38421e2-5c20-4734-bdf0-8ab3b8c721a6", "21aa1802-ad3e-4dda-b34f-ad3526f6130b",
             "bb6bb8fb-11a0-45c0-8efd-6c0791700ea0", "0528d024-7cb5-4e15-910f-39fb74b68625",
             "0a9f5909-7690-4ab6-b153-a4be885c29e0", "fec34c7e-2298-498e-896e-f0ce7716740d",
             "eeb90da1-04ce-4bb4-b054-f193cdc72b64", "3e1f1faa-a37e-41f2-a49c-dfae19b8f8a0",
             "8f522bbf-6e5f-4fed-ac59-25b242531305", "ed63c9ea-322d-40f7-bbf0-c75fb275c067"]

data = pd.concat([train, test])

data = data.groupby('kpi_id').apply(lambda x: x.sort_values('start_time'))

data['start_time_str'] = data['start_time'].map(lambda x: timestamp2string(x))

data = get_datetime(data, 'start_time_str', type='hour')
data = get_datetime(data, 'start_time_str', type='day')
data = get_datetime(data, 'start_time_str', type='month')
data = get_datetime(data, 'start_time_str', type='weekday')

for i in ['hour', 'day', 'month', 'weekday']:
    data[i] = data[i].astype('category')

num_cols = ['value']
cat_cols = ['hour', 'day', 'month', 'weekday']

data['weekday_hour'] = data['weekday'].astype(str) + '_' + data['hour'].astype(str)

for i in ['weekday_hour']:
    le = LabelEncoder()
    data[i] = le.fit_transform(data[i])
    data[i] = data[i].astype('category')
cat_cols.append('weekday_hour')

le = LabelEncoder()
data['kpi_id_le'] = le.fit_transform(data['kpi_id'])
data['kpi_id_le'] = data['kpi_id_encode'].astype('category')
cat_cols.append('kpi_id_le')

for f1 in ['kpi_id_encode']:
    for f2 in ['hour', 'day', 'month', 'weekday', 'weekday_hour']:
        data['{}_{}'.format(f1, f2)] = data[f1].astype(str) + '_' + data[f2].astype(str)
        cat_cols.append('{}_{}'.format(f1, f2))


data = cross_cat_num(data, cat_cols, num_cols)

# 时序特征
data['value_diff1'] = data.groupby('kpi_id')['value'].diff(1).fillna(0)
data['value_shift1'] = data.groupby('kpi_id')['value'].shift(1).fillna(0)
data['value_rate'] = (data['value'] - data['value_shift1']) / (data['value_shift1'] + 0.00001)

train = data.loc[data['label'].notnull(), :]
y = train['label'].astype(int)
test = data.loc[data['label'].isnull(), :]
sub = test[['start_time', 'end_time', 'kpi_id']]
del data
gc.collect()

train.drop(['kpi_id', 'start_time', 'end_time'], axis=1, inplace=True)
test.drop(['kpi_id', 'start_time', 'end_time'], axis=1, inplace=True)

train_train = train.loc[(train['start_time_str'] >= '2019:08:01 00:00:00') & (train['start_time_str'] <= '2019:09:15 23:00:00')]
train_valid = train.loc[(train['start_time_str'] >= '2019:09:16 00:00:00') & (train['start_time_str'] <= '2019:09:22 23:00:00')]

used_cols = [i for i in train.columns if i not in ['label', 'start_time_str']]

X_train = train_train[used_cols]
X_valid = train_valid[used_cols]
X_test = test[used_cols]
y_train = train_train['label']
y_valid = train_valid['label']

print('y_train mean: \n', y_train.mean())
print('y_valid mean: \n', y_valid.mean())

# 调用方法
psi_res = Parallel(n_jobs=4)(delayed(get_psi)(c, train, test) for c in used_cols)
psi_df = pd.concat(psi_res)
psi_used_cols = list(psi_df[psi_df['PSI'] <= 0.2]['变量名'].values)
psi_not_used_cols = list(psi_df[psi_df['PSI'] > 0.2]['变量名'].values)
print('PSI used features: \n', psi_used_cols)
print('PSI drop features: \n', psi_not_used_cols)
print('Error drop features: \n', list(set(used_cols) - set(psi_used_cols)))

X_train = X_train[psi_used_cols]
X_valid = X_valid[psi_used_cols]
X_test = X_test[psi_used_cols].copy()

useful_dict, useless_dict, useful_cols, useless_cols = auc_select(X_train, y_train, X_valid, y_valid, psi_used_cols,
                                                                  threshold=0.52)
print('AUC drop features: \n', useless_cols)

X_train = X_train[useful_cols]
X_valid = X_valid[useful_cols]
X_test = X_test[useful_cols]

col_corr = correlation(X_train, useful_dict, threshold=0.98)
print('Correlation drop features: \n', col_corr)

X_train.drop(col_corr, axis=1, inplace=True)
X_valid.drop(col_corr, axis=1, inplace=True)
X_test.drop(col_corr, axis=1, inplace=True)

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
