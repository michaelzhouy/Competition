# -*- coding: utf-8 -*-
# @Time     : 2020/8/10 15:59
# @Author   : Michael_Zhouy

import sys
import numpy as np
import pandas as pd
import os
import gc
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import LabelEncoder
import datetime
import time
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


def get_shift_feature(df_, start, end, col, group):
    """
    历史平移特征
    @param df_:
    @param start:
    @param end:
    @param col: label, popularity
    @param group: adcode_model_mt, model_mt，根据该列来shift，然后map
    @return: 新增列，例如label_adcode_model_mt_1
    """
    df = df_.copy()
    add_feat = []
    for i in range(start, end + 1):
        add_feat.append('shift_{}_{}_{}'.format(col, group, i))
        df['{}_{}'.format(col, i)] = df[group] + i
        df_last = df[df[col].notnull()].set_index('{}_{}'.format(col, i))
        df['shift_{}_{}_{}'.format(col, group, i)] = df[group].map(df_last[col])
        del df['{}_{}'.format(col, i)]
        gc.collect()

    return df, add_feat


def get_adjoin_feature(df_, start, end, col, group, space):
    """
    相邻N月的首尾统计（求和、求均值、差值、比例），shift统一为adcode_model_mt
    @param df_:
    @param start:
    @param end:
    @param col:
    @param group:
    @param space: 间隔
    @return:
    """
    df = df_.copy()
    add_feat = []
    for i in range(start, end + 1):
        add_feat.append('adjoin_{}_{}_{}_{}_{}_sum'.format(col, group, i, i + space, space))  # 求和
        add_feat.append('adjoin_{}_{}_{}_{}_{}_mean'.format(col, group, i, i + space, space))  # 均值
        add_feat.append('adjoin_{}_{}_{}_{}_{}_diff'.format(col, group, i, i + space, space))  # 首尾差值
        add_feat.append('adjoin_{}_{}_{}_{}_{}_ratio'.format(col, group, i, i + space, space))  # 首尾比例
        df['adjoin_{}_{}_{}_{}_{}_sum'.format(col, group, i, i + space, space)] = 0
        # sum需要遍历
        for j in range(0, space + 1):
            df['adjoin_{}_{}_{}_{}_{}_sum'.format(col, group, i, i + space, space)] = (df['adjoin_{}_{}_{}_{}_{}_sum'.format(col, group, i, i + space, space)]
                                                                                       + df['shift_{}_{}_{}'.format(col, 'adcode_model_mt', i + j)])
        # 平均
        df['adjoin_{}_{}_{}_{}_{}_mean'.format(col, group, i, i + space, space)] = (df['adjoin_{}_{}_{}_{}_{}_sum'.format(col, group, i, i + space, space)]
                                                                                    / (space + 1))
        # (当前i)-(space+i)
        df['adjoin_{}_{}_{}_{}_{}_diff'.format(col, group, i, i + space, space)] = (df['shift_{}_{}_{}'.format(col, 'adcode_model_mt', i)]
                                                                                    - df['shift_{}_{}_{}'.format(col, 'adcode_model_mt', i + space)])
        # (当前i)/(space+i)
        df['adjoin_{}_{}_{}_{}_{}_ratio'.format(col, group, i, i + space, space)] = (df['shift_{}_{}_{}'.format(col, 'adcode_model_mt', i)]
                                                                                     / df['shift_{}_{}_{}'.format(col, 'adcode_model_mt', i + space)])
    return df, add_feat


def get_series_feature(df_, start, end, col, group, types):
    """
    连续N月的统计值，shift统一为adcode_model_mt
    @param df_:
    @param start:
    @param end:
    @param col:
    @param group:
    @param types:
    @return:
    """
    df = df_.copy()
    add_feat = []
    li = []
    df['series_{}_{}_{}_{}_sum'.format(col, group, start, end)] = 0
    for i in range(start, end + 1):
        li.append('shift_{}_{}_{}'.format(col, 'adcode_model_mt', i))
    df['series_{}_{}_{}_{}_sum'.format(col, group, start, end)] = df[li].apply(get_sum, axis=1)
    df['series_{}_{}_{}_{}_mean'.format(col, group, start, end)] = df[li].apply(get_mean, axis=1)
    df['series_{}_{}_{}_{}_min'.format(col, group, start, end)] = df[li].apply(get_min, axis=1)
    df['series_{}_{}_{}_{}_max'.format(col, group, start, end)] = df[li].apply(get_max, axis=1)
    df['series_{}_{}_{}_{}_std'.format(col, group, start, end)] = df[li].apply(get_std, axis=1)
    df['series_{}_{}_{}_{}_ptp'.format(col, group, start, end)] = df[li].apply(get_ptp, axis=1)
    for typ in types:
        add_feat.append('series_{}_{}_{}_{}_{}'.format(col, group, start, end, typ))

    return df, add_feat


def getStatFeature(df_, month, flag=None):
    df = df_.copy()
    stat_feat = []

    # 确定起始位置
    if (month == 26) & (flag):
        n = 1
    elif (month == 27) & (flag):
        n = 2
    elif (month == 28) & (flag):
        n = 3
    else:
        n = 0
    print('进行统计的起始位置：', n, ' month:', month)

    ######################
    # 省份/车型/月份 粒度 #
    #####################
    df['adcode_model'] = df['adcode'] + df['model']
    df['adcode_model_mt'] = df['adcode_model'] * 100 + df['mt']
    for col in tqdm(['label']):
        # 平移
        start, end = 1 + n, 9
        df, add_feat = get_shift_feature(df, start, end, col, 'adcode_model_mt')
        stat_feat = stat_feat + add_feat

        # 相邻
        start, end = 1 + n, 8
        df, add_feat = get_adjoin_feature(df, start, end, col, 'adcode_model_mt', space=1)
        stat_feat = stat_feat + add_feat
        start, end = 1 + n, 7
        df, add_feat = get_adjoin_feature(df, start, end, col, 'adcode_model_mt', space=2)
        stat_feat = stat_feat + add_feat
        start, end = 1 + n, 6
        df, add_feat = get_adjoin_feature(df, start, end, col, 'adcode_model_mt', space=3)
        stat_feat = stat_feat + add_feat

        # 连续
        start, end = 1 + n, 3 + n
        df, add_feat = get_series_feature(df, start, end, col, 'adcode_model_mt', ['sum', 'mean', 'min', 'max', 'std', 'ptp'])
        stat_feat = stat_feat + add_feat
        start, end = 1 + n, 5 + n
        df, add_feat = get_series_feature(df, start, end, col, 'adcode_model_mt', ['sum', 'mean', 'min', 'max', 'std', 'ptp'])
        stat_feat = stat_feat + add_feat
        start, end = 1 + n, 7 + n
        df, add_feat = get_series_feature(df, start, end, col, 'adcode_model_mt', ['sum', 'mean', 'min', 'max', 'std', 'ptp'])
        stat_feat = stat_feat + add_feat

    for col in tqdm(['popularity']):
        # 平移
        start, end = 4, 9
        df, add_feat = get_shift_feature(df, start, end, col, 'adcode_model_mt')
        stat_feat = stat_feat + add_feat

        # 相邻
        start, end = 4, 8
        df, add_feat = get_adjoin_feature(df, start, end, col, 'adcode_model_mt', space=1)
        stat_feat = stat_feat + add_feat
        start, end = 4, 7
        df, add_feat = get_adjoin_feature(df, start, end, col, 'adcode_model_mt', space=2)
        stat_feat = stat_feat + add_feat
        start, end = 4, 6
        df, add_feat = get_adjoin_feature(df, start, end, col, 'adcode_model_mt', space=3)
        stat_feat = stat_feat + add_feat

        # 连续
        start, end = 4, 7
        df, add_feat = get_series_feature(df, start, end, col, 'adcode_model_mt',
                                          ['sum', 'mean', 'min', 'max', 'std', 'ptp'])
        stat_feat = stat_feat + add_feat
        start, end = 4, 9
        df, add_feat = get_series_feature(df, start, end, col, 'adcode_model_mt',
                                          ['sum', 'mean', 'min', 'max', 'std', 'ptp'])
        stat_feat = stat_feat + add_feat

    ##################
    # 车型/月份 粒度 #
    ##################
    df['model_mt'] = df['model'] * 100 + df['mt']
    for col in tqdm(['label']):
        colname = 'model_mt_{}'.format(col)
        tmp = df.groupby(['model_mt'])[col].agg({'mean'}).reset_index()
        tmp.columns = ['model_mt', colname]
        df = df.merge(tmp, on=['model_mt'], how='left')
        # 平移
        start, end = 1 + n, 9
        df, add_feat = get_shift_feature(df, start, end, colname, 'adcode_model_mt')
        stat_feat = stat_feat + add_feat

        # 相邻
        start, end = 1 + n, 8
        df, add_feat = get_adjoin_feature(df, start, end, colname, 'model_mt', space=1)
        stat_feat = stat_feat + add_feat
        start, end = 1 + n, 7
        df, add_feat = get_adjoin_feature(df, start, end, colname, 'model_mt', space=2)
        stat_feat = stat_feat + add_feat
        start, end = 1 + n, 6
        df, add_feat = get_adjoin_feature(df, start, end, colname, 'model_mt', space=3)
        stat_feat = stat_feat + add_feat

        # 连续
        start, end = 1 + n, 3 + n
        df, add_feat = get_series_feature(df, start, end, colname, 'model_mt', ['sum', 'mean'])
        stat_feat = stat_feat + add_feat
        start, end = 1 + n, 5 + n
        df, add_feat = get_series_feature(df, start, end, colname, 'model_mt', ['sum', 'mean'])
        stat_feat = stat_feat + add_feat
        start, end = 1 + n, 7 + n
        df, add_feat = get_series_feature(df, start, end, colname, 'model_mt', ['sum', 'mean'])
        stat_feat = stat_feat + add_feat

    return df, stat_feat


def score(data, pred='pred_label', label='label', group='model'):
    data['pred_label'] = data['pred_label'].apply(lambda x: 0 if x < 0 else x).round().astype(int)
    data_agg = data.groupby('model').agg({
        pred:  list,
        label: [list, 'mean']
    }).reset_index()
    data_agg.columns = ['_'.join(col).strip() for col in data_agg.columns]
    nrmse_score = []
    for raw in data_agg[['{0}_list'.format(pred), '{0}_list'.format(label), '{0}_mean'.format(label)]].values:
        nrmse_score.append(
            mse(raw[0], raw[1]) ** 0.5 / raw[2]
        )
    print(1 - np.mean(nrmse_score))
    return 1 - np.mean(nrmse_score)


for month in [25, 26, 27, 28]:

    m_type = 'xgb'
    flag = False  # False:连续提取  True:跳跃提取
    st = 4  # 保留训练集起始位置

    # 提取特征
    data_df, stat_feat = getStatFeature(data, month, flag)

    # 特征分类
    num_feat = ['regYear'] + stat_feat
    cate_feat = ['adcode', 'bodyType', 'model', 'regMonth']

    # 类别特征处理
    if m_type == 'lgb':
        for i in cate_feat:
            data_df[i] = data_df[i].astype('category')
    elif m_type == 'xgb':
        lbl = LabelEncoder()
        for i in tqdm(cate_feat):
            data_df[i] = lbl.fit_transform(data_df[i].astype(str))

    # 最终特征集
    features = num_feat + cate_feat
    print(len(features), len(set(features)))

    # 模型训练
    sub, model = get_train_model(data_df, month, m_type, st)

    data.loc[(data.regMonth == (month - 24)) & (data.regYear == 2018), 'salesVolume'] = sub['forecastVolum'].values
    data.loc[(data.regMonth == (month - 24)) & (data.regYear == 2018), 'label'] = sub['forecastVolum'].values


path  = './ccf_car/'
train_sales = pd.read_csv(path+'train_sales_data.csv')
train_search = pd.read_csv(path+'train_search_data.csv')
train_user = pd.read_csv(path+'train_user_reply_data.csv')
evaluation_public = pd.read_csv(path+'evaluation_public.csv')
submit_example = pd.read_csv(path+'submit_example.csv')
data = pd.concat([train_sales, evaluation_public], ignore_index=True)
data = data.merge(train_search, 'left', on=['province', 'adcode', 'model', 'regYear', 'regMonth'])
data = data.merge(train_user, 'left', on=['model', 'regYear', 'regMonth'])
data['label'] = data['salesVolume']
data['id'] = data['id'].fillna(0).astype(int)
data['bodyType'] = data['model'].map(train_sales.drop_duplicates('model').set_index('model')['bodyType'])
# LabelEncoder
for i in ['bodyType', 'model']:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(data[i].nunique()))))
data['mt'] = (data['regYear'] - 2016) * 12 + data['regMonth']


def get_stat_feature(df_):
    df = df_.copy()
    stat_feat = []
    # uid
    df['model_adcode'] = df['adcode'] + df['model']
    df['model_adcode_mt'] = df['model_adcode'] * 100 + df['mt']
    for col in tqdm(['label', 'popularity']):
        # shift
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            stat_feat.append('shift_model_adcode_mt_{}_{}'.format(col, i))
            df['model_adcode_mt_{}_{}'.format(col, i)] = df['model_adcode_mt'] + i
            df_last = df[df[col].notnull()].set_index('model_adcode_mt_{}_{}'.format(col, i))
            df['shift_model_adcode_mt_{}_{}'.format(col, i)] = df['model_adcode_mt'].map(df_last[col])
    return df, stat_feat


def get_model_type(train_x, train_y, valid_x, valid_y, m_type='lgb'):
    if m_type == 'lgb':
        model = lgb.LGBMRegressor(
            num_leaves=2 ** 5 - 1,
            reg_alpha=0.25,
            reg_lambda=0.25,
            objective='mse',
            max_depth=-1,
            learning_rate=0.05,
            min_child_samples=5,
            random_state=2019,
            n_estimators=2000,
            subsample=0.9,
            colsample_bytree=0.7
        )

        model.fit(
            train_x, train_y,
            eval_set=[(train_x, train_y), (valid_x, valid_y)],
            categorical_feature=cate_feat,
            early_stopping_rounds=100,
            verbose=100
        )
    elif m_type == 'xgb':
        model = xgb.XGBRegressor(
            max_depth=5,
            learning_rate=0.05,
            n_estimators=2000,
            objective='reg:gamma',
            tree_method='hist',
            subsample=0.9,
            colsample_bytree=0.7,
            min_child_samples=5,
            eval_metric='rmse'
        )
        model.fit(
            train_x, train_y,
            eval_set=[(train_x, train_y), (valid_x, valid_y)],
            early_stopping_rounds=100,
            verbose=100
        )
    return model


def get_train_model(df_, m, m_type='lgb'):
    df = df_.copy()
    # 数据集划分
    st = 13
    all_idx = (df['mt'].between(st, m - 1))
    train_idx = (df['mt'].between(st, m - 5))
    valid_idx = (df['mt'].between(m - 4, m - 4))
    test_idx = (df['mt'].between(m, m))
    print('all_idx  :', st, m - 1)
    print('train_idx:', st, m - 5)
    print('valid_idx:', m - 4, m - 4)
    print('test_idx :', m, m)
    # 最终确认
    train_x = df[train_idx][features]
    train_y = df[train_idx]['label']
    valid_x = df[valid_idx][features]
    valid_y = df[valid_idx]['label']
    # get model
    model = get_model_type(train_x, train_y, valid_x, valid_y, m_type)
    # offline，线下验证
    df['pred_label'] = model.predict(df[features])
    best_score = score(df[valid_idx])

    # online，线上预测
    if m_type == 'lgb':
        model.n_estimators = model.best_iteration_ + 100
        model.fit(df[all_idx][features], df[all_idx]['label'], categorical_feature=cate_feat)
    elif m_type == 'xgb':
        model.n_estimators = model.best_iteration + 100
        model.fit(df[all_idx][features], df[all_idx]['label'])
    df['forecastVolum'] = model.predict(df[features])
    print('valid mean:', df[valid_idx]['pred_label'].mean())
    print('true  mean:', df[valid_idx]['label'].mean())
    print('test  mean:', df[test_idx]['forecastVolum'].mean())

    # 阶段结果
    sub = df[test_idx][['id']]
    sub['forecastVolum'] = df[test_idx]['forecastVolum'].apply(lambda x: 0 if x < 0 else x).round().astype(int)
    return sub, df[valid_idx]['pred_label']


for month in [25, 26, 27, 28]:
    m_type = 'lgb'

    data_df, stat_feat = get_stat_feature(data)

    num_feat = ['regYear'] + stat_feat
    cate_feat = ['adcode', 'bodyType', 'model', 'regMonth']

    if m_type == 'lgb':
        for i in cate_feat:
            data_df[i] = data_df[i].astype('category')
    elif m_type == 'xgb':
        lbl = LabelEncoder()
        for i in tqdm(cate_feat):
            data_df[i] = lbl.fit_transform(data_df[i].astype(str))

    features = num_feat + cate_feat
    print(len(features), len(set(features)))

    sub, val_pred = get_train_model(data_df, month, m_type)
    data.loc[(data.regMonth == (month - 24)) & (data.regYear == 2018), 'salesVolume'] = sub['forecastVolum'].values
    data.loc[(data.regMonth == (month - 24)) & (data.regYear == 2018), 'label'] = sub['forecastVolum'].values

sub = data.loc[(data.regMonth >= 1) & (data.regYear == 2018), ['id', 'salesVolume']]
sub.columns = ['id', 'forecastVolum']
sub[['id', 'forecastVolum']].round().astype(int).to_csv('CCF_sales.csv', index=False)
