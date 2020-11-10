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


def auc_select(X_train, y_train, X_valid, y_valid, cols, threshold=0.52):
    """
    基于AUC的单特征筛选
    @param X_train:
    @param y_train:
    @param X_valid:
    @param y_valid:
    @param cols:
    @param threshold:
    @return:
    """
    useful_dict = dict()
    useless_dict = dict()
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
    for i in cols:
        print(i)
        try:
            lgb_train = lgb.Dataset(X_train[[i]].values, y_train)
            lgb_valid = lgb.Dataset(X_valid[[i]].values, y_valid, reference=lgb_train)
            lgb_model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_valid, lgb_train],
                num_boost_round=1000,
                early_stopping_rounds=50,
                verbose_eval=500
            )
            print('*' * 10)
            print(lgb_model.best_score['valid_0']['auc'])
            if lgb_model.best_score['valid_0']['auc'] > threshold:
                useful_dict[i] = lgb_model.best_score['valid_0']['auc']
            else:
                useless_dict[i] = lgb_model.best_score['valid_0']['auc']
        except:
            print('Error: ', i)
    useful_cols = list(useful_dict.keys())
    useless_cols = list(useless_dict.keys())
    return useful_dict, useless_dict, useful_cols, useless_cols


def correlation(df, useful_dict, threshold=0.98):
    """
    去除特征相关系数大于阈值的特征
    @param df:
    @param threshold:
    @param useful_dict:
    @return:
    """
    col_corr = set()
    corr_matrix = df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colName_i = corr_matrix.columns[i]
                colName_j = corr_matrix.columns[j]
                if useful_dict[colName_i] >= useful_dict[colName_j]:
                    col_corr.add(colName_j)
                else:
                    col_corr.add(colName_i)
    return list(col_corr)


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
sub = test[['start_time', 'end_time', 'kpi_id']]
train['label'] = train['label'].astype(int)

train.sort_values(['kpi_id', 'start_time'], inplace=True)
test.sort_values(['kpi_id', 'start_time'], inplace=True)

data = pd.concat([train, test], axis=0, ignore_index=True)
data['kpi_id_num'] = data['kpi_id'].map(dict(zip(data['kpi_id'].unique(), range(data['kpi_id'].nunique()))))
data.sort_values(['kpi_id', 'start_time'], inplace=True)

data['start_time_str'] = data['start_time'].map(lambda x: timestamp2string(x))

# data = get_datetime(data, 'start_time_str', type='hour')
# data = get_datetime(data, 'start_time_str', type='day')
# data = get_datetime(data, 'start_time_str', type='month')
# data = get_datetime(data, 'start_time_str', type='weekday')
data['start_time_dt'] = pd.to_datetime(data['start_time_str'])

# data['mdh'] = ((data['month'] - 8) * 31 + data['day']) * 24 + data['hour']

for i in tqdm(range(1, 11)):
    data['shift{}'.format(i)] = data.groupby('kpi_id')['value'].shift(i)
    data['diff{}'.format(i)] = data.groupby('kpi_id')['value'].diff(i)
    data['rate{}'.format(i)] = data['value'] / (data['shift{}'.format(i)] + 0.0001) - 1

for i in tqdm([-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]):
    data['shift{}'.format(i)] = data.groupby('kpi_id')['value'].shift(i)
    data['diff{}'.format(i)] = data.groupby('kpi_id')['value'].diff(i)
    data['rate{}'.format(i)] = data['value'] / (data['shift{}'.format(i)] + 0.0001) - 1

# data.set_index('start_time_dt', inplace=True)
# print(data.index)

# for i in tqdm(range(3, 14)):
#     data['rolling{}_mean'.format(i)] = data.groupby('kpi_id')['value'].rolling(window=i).mean()
#     data['rolling{}_max'.format(i)] = data.groupby('kpi_id')['value'].rolling(window=i).max()
#     data['rolling{}_min'.format(i)] = data.groupby('kpi_id')['value'].rolling(window=i).min()
#     data['rolling{}_median'.format(i)] = data.groupby('kpi_id')['value'].rolling(window=i).median()

# data.reset_index(inplace=True)

# sp_kpi = [1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17]


train = data.loc[data['label'].notnull(), :]
test = data.loc[data['label'].isnull(), :]

######################################################
# 规则
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

i = 3
df = train.loc[train['kpi_id_num'] == i, :]
df_test = test.loc[test['kpi_id_num'] == i, :]
sub3 = df_test[['start_time', 'end_time', 'kpi_id']]
sub3['label'] = np.where(df_test['value'] < 60, 1, 0)

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

subs = pd.concat([sub0, sub2, sub3, sub4, sub13, sub18, sub19], axis=0, ignore_index=True)
#######################################################################


for kpi in [9, 12, 17]:
    if kpi == 9:
        df0 = data.loc[data['kpi_id_num'] == kpi, :]
    else:
        df1 = data.loc[data['kpi_id_num'] == kpi, :]
        df0 = pd.concat([df0, df1])

df1 = data.loc[data['kpi_id_num'] == 1, :]
df5 = data.loc[data['kpi_id_num'] == 5, :]
df6 = data.loc[data['kpi_id_num'] == 6, :]
df7 = data.loc[data['kpi_id_num'] == 7, :]
df8 = data.loc[data['kpi_id_num'] == 8, :]
df10 = data.loc[data['kpi_id_num'] == 10, :]
df11 = data.loc[data['kpi_id_num'] == 11, :]
df14 = data.loc[data['kpi_id_num'] == 14, :]
df15 = data.loc[data['kpi_id_num'] == 15, :]
df16 = data.loc[data['kpi_id_num'] == 16, :]

for df in [df0, df1, df5, df6, df7, df8, df10, df11, df14, df15, df16]:
    print(df['kpi_id_num'].unique())
    train = df.loc[df['label'].notnull(), :]
    y = train['label'].astype(int)
    test = df.loc[df['label'].isnull(), :]
    sub1 = test[['start_time', 'end_time', 'kpi_id']]
    del df
    gc.collect()

    train.drop(['kpi_id', 'start_time', 'end_time', 'kpi_id_num', 'start_time_dt'], axis=1, inplace=True)
    test.drop(['kpi_id', 'start_time', 'end_time', 'kpi_id_num', 'start_time_dt'], axis=1, inplace=True)
    gc.collect()

    train_train = train.loc[(train['start_time_str'] >= '2019-08-01 00:00:00') & (train['start_time_str'] <= '2019-09-15 23:00:00'), :]
    train_valid = train.loc[(train['start_time_str'] >= '2019-09-16 00:00:00') & (train['start_time_str'] <= '2019-09-22 23:00:00'), :]

    # print('train_train.shape: \n', train_train.shape)
    # print('train_valid.shape: \n', train_valid.shape)
    # print('test.shape: \n', test.shape)

    used_cols = [i for i in train_train.columns if i not in ['label', 'start_time_str']]

    # print('len(used_cols): \n', len(used_cols))
    # print('used_cols:\n', used_cols)
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
        'metric': 'None',
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

    def lgb_f1_score(y_hat, data):
        y_true = data.get_label()
        y_hat = np.where(y_hat > 0.5, 1, 0)
        return 'f1', f1_score(y_true, y_hat), True

    valid_model = lgb.train(
        params,
        train_dataset,
        valid_sets=[valid_dataset, train_dataset],
        early_stopping_rounds=100,
        num_boost_round=1000000,
        verbose_eval=100,
        feval=lgb_f1_score)
    pred = valid_model.predict(X_valid)
    y_valid_pred = np.where(pred > 0.5, 1, 0)
    print('Valid F1: ', np.round(f1_score(y_valid, y_valid_pred), 5))
    print('Valid mean label: ', np.mean(y_valid_pred))
    train_model = lgb.train(
        params,
        all_dataset,
        num_boost_round=valid_model.best_iteration + 10,
        feval=lgb_f1_score)
    y_test_pred = np.where(train_model.predict(X_test) > 0.5, 1, 0)
    print('Test mean label: ', np.mean(y_test_pred))
    sub1['label'] = y_test_pred

    subs = pd.concat([subs, sub1], axis=0, ignore_index=True)

sub = sub.merge(subs, on=['start_time', 'end_time', 'kpi_id'], how='left')
print('Test mean label: ', np.mean(sub['label']))
sub['label'] = sub['label'].astype(int)
print('Done')
out_path = os.path.join(Context.get_output_path(), "result" + '.csv')
with open(out_path, 'w') as f:
    sub.to_csv(f, index=False)