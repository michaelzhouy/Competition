# -*- coding: utf-8 -*-
# @Time     : 2020/7/15 14:13
# @Author   : Michael_Zhouy

import pandas as pd
import os
import gc
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings('ignore')


def get_psi(train,test,f_cols):
    psi_res = pd.DataFrame()
    psi_dict={}
    for c in tqdm(f_cols):
        try:
            t_train = train[c].fillna(-998)
            t_test = test[c].fillna(-998)
            #获取切分点
            bins=[]
            for i in np.arange(0,1.1,0.2):
                bins.append(t_train.quantile(i))
            bins=sorted(set(bins))
            bins[0]=-np.inf
            bins[-1]=np.inf
            #计算psi
            t_psi = pd.DataFrame()
            t_psi['train'] = pd.cut(t_train,bins).value_counts().sort_index()
            t_psi['test'] = pd.cut(t_test,bins).value_counts()
            if c == 'outdoorTemp':
                print(t_psi['train'])
                print(t_psi['test'])
            t_psi.index=[str(x) for x in t_psi.index]
            t_psi.loc['总计',:] = t_psi.sum()
            t_psi['train_rate'] = t_psi['train']/t_psi.loc['总计','train']
            t_psi['test_rate'] = t_psi['test']/t_psi.loc['总计','test']
            t_psi['psi'] = (t_psi['test_rate']-t_psi['train_rate'])*(np.log(t_psi['test_rate'])-np.log(t_psi['train_rate']))
            t_psi.loc['总计','psi'] = t_psi['psi'].sum()
            t_psi.index.name=c
            #汇总
            t_res = pd.DataFrame([[c,t_psi.loc['总计','psi']]],
                                 columns=['变量名','PSI'])
            psi_res = pd.concat([psi_res,t_res])
            psi_dict[c]=t_psi
            print(c,'done')
        except:
            print(c,'error')
    return psi_res,psi_dict


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
sub = pd.DataFrame(test_df['time'])

train_df = train_df[train_df['temperature'].notnull()]
train_df = train_df.fillna(method='bfill')
test_df = test_df.fillna(method='bfill')

train_df.columns = ['time', 'year', 'month', 'day', 'hour', 'min', 'sec', 'outdoorTemp', 'outdoorHum', 'outdoorAtmo',
                    'indoorHum', 'indoorAtmo', 'temperature']
test_df.columns = ['time', 'year', 'month', 'day', 'hour', 'min', 'sec', 'outdoorTemp', 'outdoorHum', 'outdoorAtmo',
                   'indoorHum', 'indoorAtmo']
print('train_df.shape: ', train_df.shape)
train_df = train_df.loc[(train_df['outdoorTemp'] >= test_df['outdoorTemp'].min()) & (train_df['outdoorTemp'] <= test_df['outdoorTemp'].max())]
print('处理后 train_df.shape: ', train_df.shape)
data_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# data_df['indoorAtmo-outdoorAtmo'] = data_df['indoorAtmo'] - data_df['outdoorAtmo']
# data_df['indoorHum-outdoorHum'] = data_df['indoorHum'] - data_df['outdoorHum']

# 基本聚合特征
group_feats = []
for f in tqdm(['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo']):
    data_df['MDH_{}_medi'.format(f)] = data_df.groupby(['month', 'day', 'hour'])[f].transform('median')
    data_df['MDH_{}_mean'.format(f)] = data_df.groupby(['month', 'day', 'hour'])[f].transform('mean')
    data_df['MDH_{}_max'.format(f)] = data_df.groupby(['month', 'day', 'hour'])[f].transform('max')
    data_df['MDH_{}_min'.format(f)] = data_df.groupby(['month', 'day', 'hour'])[f].transform('min')
    data_df['MDH_{}_std'.format(f)] = data_df.groupby(['month', 'day', 'hour'])[f].transform('std')

    group_feats.append('MDH_{}_medi'.format(f))
    group_feats.append('MDH_{}_mean'.format(f))

# 基本交叉特征
for f1 in tqdm(['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo'] + group_feats):
    for f2 in ['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo'] + group_feats:
        if f1 != f2:
            colname = '{}_{}_ratio'.format(f1, f2)
            data_df[colname] = data_df[f1].values / data_df[f2].values

data_df = data_df.fillna(method='bfill')

# 历史信息提取
data_df['dt'] = data_df['day'].values + (data_df['month'].values - 3) * 31

for f in ['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo', 'temperature']:
    tmp_df = pd.DataFrame()
    for t in tqdm(range(15, 45)):
        tmp = data_df[data_df['dt'] < t].groupby(['hour'])[f].agg({'mean'}).reset_index()
        tmp.columns = ['hour', 'hit_{}_mean'.format(f)]
        tmp['dt'] = t
        tmp_df = tmp_df.append(tmp)

    data_df = data_df.merge(tmp_df, on=['dt', 'hour'], how='left')

data_df = data_df.fillna(method='bfill')

# 离散化
for f in ['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo']:
    data_df[f + '_20_bin'] = pd.cut(data_df[f], 20, duplicates='drop').apply(lambda x: x.left).astype(int)
    data_df[f + '_50_bin'] = pd.cut(data_df[f], 50, duplicates='drop').apply(lambda x: x.left).astype(int)
    data_df[f + '_100_bin'] = pd.cut(data_df[f], 100, duplicates='drop').apply(lambda x: x.left).astype(int)
    data_df[f + '_200_bin'] = pd.cut(data_df[f], 200, duplicates='drop').apply(lambda x: x.left).astype(int)

for f1 in tqdm(
        ['outdoorTemp_20_bin', 'outdoorHum_20_bin', 'outdoorAtmo_20_bin', 'indoorHum_20_bin', 'indoorAtmo_20_bin']):
    for f2 in ['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo']:
        data_df['{}_{}_medi'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('median')
        data_df['{}_{}_mean'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('mean')
        data_df['{}_{}_max'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('max')
        data_df['{}_{}_min'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('min')

for f1 in tqdm(
        ['outdoorTemp_50_bin', 'outdoorHum_50_bin', 'outdoorAtmo_50_bin', 'indoorHum_50_bin', 'indoorAtmo_50_bin']):
    for f2 in ['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo']:
        data_df['{}_{}_medi'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('median')
        data_df['{}_{}_mean'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('mean')
        data_df['{}_{}_max'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('max')
        data_df['{}_{}_min'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('min')

for f1 in tqdm(['outdoorTemp_100_bin', 'outdoorHum_100_bin', 'outdoorAtmo_100_bin', 'indoorHum_100_bin',
                'indoorAtmo_100_bin']):
    for f2 in ['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo']:
        data_df['{}_{}_medi'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('median')
        data_df['{}_{}_mean'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('mean')
        data_df['{}_{}_max'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('max')
        data_df['{}_{}_min'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('min')

for f1 in tqdm(['outdoorTemp_200_bin', 'outdoorHum_200_bin', 'outdoorAtmo_200_bin', 'indoorHum_200_bin',
                'indoorAtmo_200_bin']):
    for f2 in ['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo']:
        data_df['{}_{}_medi'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('median')
        data_df['{}_{}_mean'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('mean')
        data_df['{}_{}_max'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('max')
        data_df['{}_{}_min'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('min')


def single_model(clf, train_x, train_y, test_x, clf_name, class_num=1):
    train = np.zeros((train_x.shape[0], class_num))
    test = np.zeros((test_x.shape[0], class_num))

    nums = int(train_x.shape[0] * 0.80)

    if clf_name in ['sgd', 'ridge']:
        print('MinMaxScaler...')
        for col in features:
            ss = MinMaxScaler()
            ss.fit(np.vstack([train_x[[col]].values, test_x[[col]].values]))
            train_x[col] = ss.transform(train_x[[col]].values).flatten()
            test_x[col] = ss.transform(test_x[[col]].values).flatten()

    trn_x, trn_y, val_x, val_y = train_x[:nums], train_y[:nums], train_x[nums:], train_y[nums:]

    if clf_name == "lgb":
        train_matrix = clf.Dataset(trn_x, label=trn_y)
        valid_matrix = clf.Dataset(val_x, label=val_y)
        data_matrix = clf.Dataset(train_x, label=train_y)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'mse',
            'min_child_weight': 5,
            'num_leaves': 2 ** 8,
            'feature_fraction': 0.5,
            'bagging_fraction': 0.5,
            'bagging_freq': 1,
            'learning_rate': 0.001,
            'seed': 2020
        }

        model = clf.train(params, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix], verbose_eval=500,
                          early_stopping_rounds=1000)
        model2 = clf.train(params, data_matrix, model.best_iteration)
        val_pred = model.predict(val_x, num_iteration=model2.best_iteration).reshape(-1, 1)
        test_pred = model.predict(test_x, num_iteration=model2.best_iteration).reshape(-1, 1)

    if clf_name == "xgb":
        train_matrix = clf.DMatrix(trn_x, label=trn_y, missing=np.nan)
        valid_matrix = clf.DMatrix(val_x, label=val_y, missing=np.nan)
        test_matrix = clf.DMatrix(test_x, label=val_y, missing=np.nan)
        params = {'booster': 'gbtree',
                  'eval_metric': 'mae',
                  'min_child_weight': 5,
                  'max_depth': 8,
                  'subsample': 0.5,
                  'colsample_bytree': 0.5,
                  'eta': 0.001,
                  'seed': 2020,
                  'nthread': 36,
                  'silent': True,
                  }

        watchlist = [(train_matrix, 'train'), (valid_matrix, 'eval')]

        model = clf.train(params, train_matrix, num_boost_round=50000, evals=watchlist, verbose_eval=500,
                          early_stopping_rounds=1000)
        val_pred = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit).reshape(-1, 1)
        test_pred = model.predict(test_matrix, ntree_limit=model.best_ntree_limit).reshape(-1, 1)

    if clf_name == "cat":
        params = {'learning_rate': 0.001, 'depth': 5, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli',
                  'od_type': 'Iter', 'od_wait': 50, 'random_seed': 11, 'allow_writing_files': False}

        model = clf(iterations=20000, **params)
        model.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                  cat_features=[], use_best_model=True, verbose=500)

        val_pred = model.predict(val_x)
        test_pred = model.predict(test_x)

    if clf_name == "sgd":
        params = {
            'loss': 'squared_loss',
            'penalty': 'l2',
            'alpha': 0.00001,
            'random_state': 2020,
        }
        model = SGDRegressor(**params)
        model.fit(trn_x, trn_y)
        val_pred = model.predict(val_x)
        test_pred = model.predict(test_x)

    if clf_name == "ridge":
        params = {
            'alpha': 1.0,
            'random_state': 2020,
        }
        model = Ridge(**params)
        model.fit(trn_x, trn_y)
        val_pred = model.predict(val_x)
        test_pred = model.predict(test_x)

    print("%s_mse_score:" % clf_name, mean_squared_error(val_y, val_pred))

    return val_pred, test_pred


def lgb_model(x_train, y_train, x_valid):
    lgb_train, lgb_test = single_model(lgb, x_train, y_train, x_valid, "lgb", 1)
    return lgb_train, lgb_test


def xgb_model(x_train, y_train, x_valid):
    xgb_train, xgb_test = single_model(xgb, x_train, y_train, x_valid, "xgb", 1)
    return xgb_train, xgb_test


def cat_model(x_train, y_train, x_valid):
    cat_train, cat_test = single_model(CatBoostRegressor, x_train, y_train, x_valid, "cat", 1)
    return cat_train, cat_test


def sgd_model(x_train, y_train, x_valid):
    sgd_train, sgd_test = single_model(SGDRegressor, x_train, y_train, x_valid, "sgd", 1)
    return sgd_train, sgd_test


def ridge_model(x_train, y_train, x_valid):
    ridge_train, ridge_test = single_model(Ridge, x_train, y_train, x_valid, "ridge", 1)
    return ridge_train, ridge_test


drop_columns = ["time", "year", "sec", "temperature"]

train_count = train_df.shape[0]
train_df = data_df[:train_count].copy().reset_index(drop=True)
test_df = data_df[train_count:].copy().reset_index(drop=True)
del data_df
gc.collect()

features = train_df[:1].drop(drop_columns, axis=1).columns
x_train = train_df[features]
x_test = test_df[features]

y_train = train_df['temperature'].values - train_df['outdoorTemp'].values

print('-' * 10)
psi_res, psi_dict = get_psi(x_train,x_test,features)
# print('-' * 10)
# print(psi_res)
# print('-' * 10)
# print(psi_dict)

features = list(psi_res[psi_res['PSI'] <= 0.2]['变量名'].values) + ['outdoorTemp']

# lr_train, lr_test = ridge_model(x_train, y_train, x_test)
#
# sgd_train, sgd_test = sgd_model(x_train, y_train, x_test)
#
# lgb_train, lgb_test = lgb_model(x_train, y_train, x_test)
#
xgb_train, xgb_test = xgb_model(x_train, y_train, x_test)
#
# cat_train, cat_test = cat_model(x_train, y_train, x_test)
#
# train_pred = (lr_train + sgd_train + lgb_train[:, 0] + xgb_train[:, 0] + cat_train) / 5
# test_pred = (lr_test + sgd_test + lgb_test[:, 0] + xgb_test[:, 0] + cat_test) / 5
#
sub["temperature"] = xgb_test[:, 0] + test_df['outdoorTemp'].values
sub.to_csv('../sub/sub_psi.csv', index=False)
