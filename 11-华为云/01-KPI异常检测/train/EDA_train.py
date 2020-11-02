import time
import os
import numpy as np
import datetime
import pandas as pd
import gc
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from naie.context import Context as context
from naie.datasets import data_reference
from naie.feature_processing import data_flow
from naie.feature_analysis import data_analysis
from naie.datasets import get_data_reference
from naie.context import Context
from naie.feature_processing.expression import col, cols, cond, f_and, f_not, f_or
from naie.common.data.typedefinition import StepType, ColumnRelationship, JoinType, ColumnSelector,DynamicColumnsSelectorDetails, StaticColumnsSelectorDetails, ColumnsSelectorDetails, DataProcessMode

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
    if type == 'hour':
        df['hour'] = df[time_col].map(lambda x: int(str(x)[11: 13]))
    elif type == 'day':
        df['day'] = df[time_col].map(lambda x: int(str(x)[8: 10]))
    elif type == 'month':
        df['month'] = df[time_col].map(lambda x: int(str(x)[5: 7]))
    elif type == 'year':
        df['year'] = df[time_col].map(lambda x: int(str(x)[0: 4]))
    elif type == 'weekday':
        df['weekday'] = pd.to_datetime(df[time_col]).map(lambda x: x.weekday())
    return df


print(time.strftime('%Y%m%d'))
train = get_data_reference(dataset='data', dataset_entity='train').to_pandas_dataframe()
test = get_data_reference(dataset='data', dataset_entity='test').to_pandas_dataframe()

train = train.groupby('kpi_id', as_index=False).apply(lambda x: x.sort_values('start_time'))
test = test.groupby('kpi_id', as_index=False).apply(lambda x: x.sort_values('start_time'))
sub = test[['start_time', 'end_time', 'kpi_id']]


data = pd.concat([train, test])
data['kpi_id_num'] = data['kpi_id'].map(dict(zip(data['kpi_id'].unique(), range(data['kpi_id'].nunique()))))
data = data.groupby('kpi_id', as_index=False).apply(lambda x: x.sort_values('start_time'))
# data['kpi_id_num'].unique()

data['start_time_str'] = data['start_time'].map(lambda x: timestamp2string(x))

# data = get_datetime(data, 'start_time_str', type='hour')
# data = get_datetime(data, 'start_time_str', type='day')
# data = get_datetime(data, 'start_time_str', type='month')
# data = get_datetime(data, 'start_time_str', type='weekday')
data['start_time_dt'] = pd.to_datetime(data['start_time_str'])

train = data.loc[data['label'].notnull(), :]
test = data.loc[data['label'].isnull(), :]

sp_kpi = [1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17]

i = 0
df = train.loc[train['kpi_id_num'] == i, :]
df_test = test.loc[test['kpi_id_num'] == i, :]
sub0 = df_test[['start_time', 'end_time', 'kpi_id']]
sub0['label'] = 0

i = 2
df = train.loc[train['kpi_id_num'] == i, :]
df_test = test.loc[test['kpi_id_num'] == i, :]
sub2 = df_test[['start_time', 'end_time', 'kpi_id']]
sub2['label'] = np.where(df_test['value'] < 90, 1, 0)

i = 4
df = train.loc[train['kpi_id_num'] == i, :]
df_test = test.loc[test['kpi_id_num'] == i, :]
sub4 = df_test[['start_time', 'end_time', 'kpi_id']]
sub4['label'] = 0

i = 13
df = train.loc[train['kpi_id_num'] == i, :]
df_test = test.loc[test['kpi_id_num'] == i, :]
sub13 = df_test[['start_time', 'end_time', 'kpi_id']]
sub13['label'] = np.where(df_test['value'] > 1000, 1, 0)

i = 18
df = train.loc[train['kpi_id_num'] == i, :]
df_test = test.loc[test['kpi_id_num'] == i, :]
sub18 = df_test[['start_time', 'end_time', 'kpi_id']]
sub18['label'] = np.where(df_test['value'] > 1000, 1, 0)

i = 19
df = train.loc[train['kpi_id_num'] == i, :]
df_test = test.loc[test['kpi_id_num'] == i, :]
sub19 = df_test[['start_time', 'end_time', 'kpi_id']]
sub19['label'] = np.where(df_test['value'] < 10000, 1, 0)

subs = pd.concat([sub0, sub2, sub4, sub13, sub18, sub19])

for kpi in sp_kpi:
    df = data.loc[data['kpi_id_num'] == kpi, :]
    df.sort_values('start_time', inplace=True)
    for i in tqdm([1, 2, 3]):
        df['shift{}'.format(i)] = df['value'].shift(i)
        df['diff{}'.format(i)] = df['value'].diff(i)
    df['div1'] = (df['value'] - df['shift1']) / (df['shift1'] + 0.001)
    df['div2'] = (df['value'] - df['shift2']) / (df['shift2'] + 0.001)
    df['div3'] = (df['value'] - df['shift3']) / (df['shift3'] + 0.001)
    train = df.loc[df['label'].notnull(), :]
    test = df.loc[df['label'].isnull(), :]
    submission = test[['start_time', 'end_time', 'kpi_id']]
    y = train['label']

    train_train = train.loc[(train['start_time_str'] >= '2019-08-01 00:00:00') & (train['start_time_str'] <= '2019-09-15 23:00:00'), :]
    train_valid = train.loc[(train['start_time_str'] >= '2019-09-16 00:00:00') & (train['start_time_str'] <= '2019-09-22 23:00:00'), :]

    print('train_train.shape: \n', train_train.shape)
    print('train_valid.shape: \n', train_valid.shape)
    print('test.shape: \n', test.shape)
    
    used_cols = [col for col in train.columns if col not in ['start_time', 'end_time', 'kpi_id', 'kpi_id_num', 'start_time_str', 'start_time_dt']]
    print('used_cols:\n', used_cols)
    X_train = train_train[used_cols]
    X_valid = train_valid[used_cols]
    X_test = test[used_cols]
    y_train = train_train['label']
    y_valid = train_valid['label']
    print('y_train mean: \n', y_train.mean())
    print('y_valid mean: \n', y_valid.mean())

    train_dataset = lgb.Dataset(X_train, y_train)
    valid_dataset = lgb.Dataset(X_valid, y_valid, reference=train_dataset)
    all_dataset = lgb.Dataset(train[used_cols], y, reference=train_dataset)

    params = {
        'objective': 'binary',
        'boosting': 'gbdt',
        'metric': 'auc',
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
        early_stopping_rounds=100,
        num_boost_round=1000000,
        verbose_eval=100
    )

    pred = valid_model.predict(X_valid)

    y_valid_pred = np.where(pred > 0.5, 1, 0)
    print('Valid F1: ', np.round(f1_score(y_valid, y_valid_pred), 5))
    print('Valid mean label: ', np.mean(y_valid_pred))

    train_model = lgb.train(
        params,
        all_dataset,
        num_boost_round=valid_model.best_iteration + 10
    )

    y_test_pred = np.where(train_model.predict(X_test) > 0.5, 1, 0)

    print('Test mean label: ', np.mean(y_test_pred))
    submission['label'] = y_test_pred
    subs = pd.concat([subs, submission])
sub = sub.merge(subs, on=['start_time', 'end_time', 'kpi_id'], how='left')
out_path = os.path.join(Context.get_output_path(), "result" + '.csv')
with open(out_path, 'w') as f:
    sub.to_csv(f, index=False)