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


def df_plot(df_train, df_test):
    plt.figure(figsize=(25, 10))
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(df_train['start_time_dt'], df_train['value'])
    ax1.plot(df_test['start_time_dt'], df_test['value'], c='g')
    size_map = {1: 30, 0: 20}
    train_size = df_train['label'].map(size_map)
    test_size = df_test['label'].map(size_map)
    color_map = {1: 'r', 0: 'b'}
    train_color = df_train['label'].map(color_map)
    color_map = {1: 'r', 0: 'g'}
    test_color = df_test['label'].map(color_map)
    ax1.scatter(df_train['start_time_dt'], df_train['value'], s=train_size, c=train_color)
    ax1.scatter(df_test['start_time_dt'], df_test['value'], s=test_size, c=test_color)
    ax1.set_ylabel('value')
    plt.show()


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


def FE(df, shift_n):
    for i in tqdm(list(range(-shift_n, shift_n + 1))):
        if i != 0:
            df['shift_{}'.format(i)] = df['value'].shift(i)
            df['diff_{}'.format(i)] = df['value'].diff(i)
            df['rate_{}'.format(i)] = df['value'] / (df['shift_{}'.format(i)] + 0.0001) - 1
    for i in [4]:
        df['shift_{}_mean'.format(i)] = df.loc[:, 'shift_-{}'.format(i):'shift_{}'.format(i)].mean(axis=1)
        df['value-shift_{}_mean'.format(i)] = df['value'] - df['shift_{}_mean'.format(i)]
    return df


def build_model(df_):
    train = df_.loc[df_['label'].notnull(), :]
    y_train = train['label'].astype(int)
    test = df_.loc[df_['label'].isnull(), :]
    sub = test[['start_time', 'end_time', 'kpi_id']]
    del df_
    gc.collect()

    mark_cols = ['label', 'start_time', 'end_time', 'kpi_id', 'kpi_id_num', 'start_time_str', 'start_time_dt']
    used_cols = [i for i in train.columns if i not in mark_cols]
    X_train = train[used_cols]
    X_test = test[used_cols]
    
    dtrain = lgb.Dataset(X_train, y_train)

    lgb_model = lgb.train(
        params,
        dtrain,
        num_boost_round=20,
        verbose_eval=5)
    
    pred = lgb_model.predict(X_test)
    y_pred = np.where(pred > 0.5, 1, 0)
    sub['label'] = y_pred
    test['label'] = y_pred
    return train, test, sub


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
    'seed': 2020}


print(time.strftime('%Y%m%d'))
train = get_data_reference(dataset='data', dataset_entity='train').to_pandas_dataframe()
test = get_data_reference(dataset='data', dataset_entity='test').to_pandas_dataframe()

submission = test[['start_time', 'end_time', 'kpi_id']]

train = train.groupby('kpi_id', as_index=False).apply(lambda x: x.sort_values('start_time'))
test = test.groupby('kpi_id', as_index=False).apply(lambda x: x.sort_values('start_time'))

data = pd.concat([train, test])
data['kpi_id_num'] = data['kpi_id'].map(dict(zip(data['kpi_id'].unique(), range(data['kpi_id'].nunique()))))
data = data.groupby('kpi_id', as_index=False).apply(lambda x: x.sort_values('start_time'))
data['kpi_id_num'].unique()

data['start_time_str'] = data['start_time'].map(lambda x: timestamp2string(x))
data['start_time_dt'] = pd.to_datetime(data['start_time_str'])

for i in range(20):
    exec("df{} = data.loc[data['kpi_id_num'] == i, :]".format(i))

subs = pd.DataFrame()

###################### df0 ###########################
df = df0.copy()

df_train = df.loc[df['label'].notnull(), :]
df_test = df.loc[df['label'].isnull(), :]
df_test['label'] = 0

sub = df_test[['start_time', 'end_time', 'kpi_id']]
sub['label'] = 0

subs = pd.concat([subs, sub], axis=0, ignore_index=True)
df_plot(df_train, df_test)
###################### df0 ###########################


###################### df2 ###########################
df = df2.copy()

df_train = df.loc[df['label'].notnull(), :]
df_test = df.loc[df['label'].isnull(), :]
df_test['label'] = np.where(df_test['value'] < 90, 1, 0)

sub = df_test[['start_time', 'end_time', 'kpi_id']]
sub['label'] = np.where(df_test['value'] < 90, 1, 0)

subs = pd.concat([subs, sub], axis=0, ignore_index=True)
df_plot(df_train, df_test)
###################### df2 ###########################


###################### df3 ###########################
df = df3.copy()

df_train = df.loc[df['label'].notnull(), :]
df_test = df.loc[df['label'].isnull(), :]
df_test['label'] = np.where(df_test['value'] < 71, 1, 0)

sub = df_test[['start_time', 'end_time', 'kpi_id']]
sub['label'] = np.where(df_test['value'] < 71, 1, 0)

subs = pd.concat([subs, sub], axis=0, ignore_index=True)
df_plot(df_train, df_test)
###################### df3 ###########################


###################### df4 ###########################
df = df4.copy()

df_train = df.loc[df['label'].notnull(), :]
df_test = df.loc[df['label'].isnull(), :]
df_test['label'] = 0

sub = df_test[['start_time', 'end_time', 'kpi_id']]
sub['label'] = 0

subs = pd.concat([subs, sub], axis=0, ignore_index=True)
df_plot(df_train, df_test)
###################### df4 ###########################


###################### df13 ###########################
df = df13.copy()

df_train = df.loc[df['label'].notnull(), :]
df_test = df.loc[df['label'].isnull(), :]
df_test['label'] = np.where(df_test['value'] > 1000, 1, 0)

sub = df_test[['start_time', 'end_time', 'kpi_id']]
sub['label'] = np.where(df_test['value'] > 1000, 1, 0)

subs = pd.concat([subs, sub], axis=0, ignore_index=True)
df_plot(df_train, df_test)
###################### df13 ###########################


###################### df18 ###########################
df = df18.copy()

df_train = df.loc[df['label'].notnull(), :]
df_test = df.loc[df['label'].isnull(), :]
df_test['label'] = np.where(df_test['value'] > 1000, 1, 0)

sub = df_test[['start_time', 'end_time', 'kpi_id']]
sub['label'] = np.where(df_test['value'] > 1000, 1, 0)

subs = pd.concat([subs, sub], axis=0, ignore_index=True)
df_plot(df_train, df_test)
###################### df18 ###########################


###################### df19 ###########################
df = df19.copy()

df_train = df.loc[df['label'].notnull(), :]
df_test = df.loc[df['label'].isnull(), :]
df_test['label'] = np.where(df_test['value'] < 10000, 1, 0)

sub = df_test[['start_time', 'end_time', 'kpi_id']]
sub['label'] = np.where(df_test['value'] < 10000, 1, 0)

subs = pd.concat([subs, sub], axis=0, ignore_index=True)
df_plot(df_train, df_test)
###################### df19 ###########################


###################### df8 ###########################
df = df8.copy()

df_train = df.loc[df['label'].notnull(), :]
df_test = df.loc[df['label'].isnull(), :]
df_test['label'] = np.where(df_test['value'] < 70, 1, 0)

sub = df_test[['start_time', 'end_time', 'kpi_id']]
sub['label'] = np.where(df_test['value'] < 70, 1, 0)

subs = pd.concat([subs, sub], axis=0, ignore_index=True)
df_plot(df_train, df_test)
###################### df8 ###########################


###################### df1 ###########################
df = df1.copy()
df.sort_values('start_time', inplace=True)
df_fe = FE(df, 35)
df_train, df_test, sub = build_model(df_fe)
subs = pd.concat([subs, sub], axis=0, ignore_index=True)
df_plot(df_train, df_test)
###################### df1 ###########################


###################### df5 ###########################
df = df5.copy()
df.sort_values('start_time', inplace=True)
df_fe = FE(df, 15)
df_train, df_test, sub = build_model(df_fe)
subs = pd.concat([subs, sub], axis=0, ignore_index=True)
df_plot(df_train, df_test)
###################### df5 ###########################


###################### df6 ###########################
df = df6.copy()
df.sort_values('start_time', inplace=True)
df_fe = FE(df, 15)
train, test, sub = build_model(df_fe)
subs = pd.concat([subs, sub], axis=0, ignore_index=True)
df_plot(df_train, df_test)
###################### df6 ###########################


###################### df7 ###########################
df = df7.copy()
df.sort_values('start_time', inplace=True)
df_fe = FE(df, 15)
train, test, sub = build_model(df_fe)
subs = pd.concat([subs, sub], axis=0, ignore_index=True)
df_plot(df_train, df_test)
###################### df7 ###########################


###################### df10 ###########################
df = df10.copy()
df.sort_values('start_time', inplace=True)
df_fe = FE(df, 15)
train, test, sub = build_model(df_fe)
subs = pd.concat([subs, sub], axis=0, ignore_index=True)
df_plot(df_train, df_test)
###################### df10 ###########################


###################### df11 ###########################
df = df11.copy()
df.sort_values('start_time', inplace=True)
df_fe = FE(df, 15)
train, test, sub = build_model(df_fe)
subs = pd.concat([subs, sub], axis=0, ignore_index=True)
df_plot(df_train, df_test)
###################### df11 ###########################


###################### df14 ###########################
df = df14.copy()
df.sort_values('start_time', inplace=True)
df_fe = FE(df, 15)
train, test, sub = build_model(df_fe)
subs = pd.concat([subs, sub], axis=0, ignore_index=True)
df_plot(df_train, df_test)
###################### df14 ###########################


###################### df15 ###########################
df = df15.copy()
df.sort_values('start_time', inplace=True)
df_fe = FE(df, 15)
train, test, sub = build_model(df_fe)
subs = pd.concat([subs, sub], axis=0, ignore_index=True)
df_plot(df_train, df_test)
###################### df15 ###########################


###################### df16 ###########################
df = df16.copy()
df.sort_values('start_time', inplace=True)
df_fe = FE(df, 15)
train, test, sub = build_model(df_fe)
subs = pd.concat([subs, sub], axis=0, ignore_index=True)
df_plot(df_train, df_test)
###################### df16 ###########################


###################### df9, df12, df17 ###########################
def FE(df):
    for i in tqdm(list(range(-15, 16))):
        if i != 0:
            df['shift_{}'.format(i)] = df.groupby('kpi_id')['value'].shift(i)
            df['diff_{}'.format(i)] = df.groupby('kpi_id')['value'].diff(i)
            df['rate_{}'.format(i)] = df['value'] / (df['shift_{}'.format(i)] + 0.0001) - 1   
    for i in range(2, 10):
        df['shift_{}_mean'.format(i)] = df.loc[:, 'shift_-{}'.format(i):'shift_{}'.format(i)].mean(axis=1)
        df['shift_{}_sum'.format(i)] = df.loc[:, 'shift_-{}'.format(i):'shift_{}'.format(i)].sum(axis=1)
        df['shift_{}_max'.format(i)] = df.loc[:, 'shift_-{}'.format(i):'shift_{}'.format(i)].max(axis=1)
        df['shift_{}_min'.format(i)] = df.loc[:, 'shift_-{}'.format(i):'shift_{}'.format(i)].min(axis=1)
        df['value-shift_{}_mean'.format(i)] = df['value'] - df['shift_{}_mean'.format(i)]
    return df


df = pd.concat([df9, df12, df17], axis=0, ignore_index=True)
df.sort_values(['kpi_id', 'start_time'], inplace=True)
df_fe = FE(df)
train, test, sub = build_model(df_fe)
subs = pd.concat([subs, sub], axis=0, ignore_index=True)
###################### df9, df12, df17 ###########################

print(subs.head())
submission = submission.merge(subs, on=['start_time', 'end_time', 'kpi_id'], how='left')

print('Test mean label: ', np.mean(submission['label']))
submission['label'] = submission['label'].astype(int)

print('Done')
out_path = os.path.join(Context.get_output_path(), "result" + '.csv')
with open(out_path, 'w') as f:
    submission.to_csv(f, index=False)