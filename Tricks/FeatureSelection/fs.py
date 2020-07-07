# -*- coding: utf-8 -*-
# @Time     : 2020/7/7 20:50
# @Author   : Michael_Zhouy

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split


def identify_single_unique(df):
    """

    Parameters
    ----------
    df

    Returns
    -------

    """
    unique_cnts = df.nunique()
    unique_cnts = unique_cnts.sort_values(by='nunique', ascending=True)
    to_drop = unique_cnts[unique_cnts == 1].index.to_list()
    print('{} features with a single unique value.\n'.format(len(to_drop)))
    return to_drop


def identify_missing(df, missing_threshold):
    """

    Parameters
    ----------
    df
    missing_threshold

    Returns
    -------

    """
    missing_rate = df.isnull().sum() / len(df)
    missing_rate = missing_rate.sort_values(ascending=False)
    to_drop = missing_rate[missing_rate > missing_threshold].index.to_list()
    print('{} features with greater than %0.2f missing values.\n'.format(len(to_drop)))
    return to_drop


def auc_select(df_train, auc_threshold=0.5):
    """
    单特征AUC筛选
    @param df_train:
    @param auc_threshold: AUC阈值
    @return:
    """
    X_train, X_valid, y_train, y_valid = train_test_split(df_train.drop(['phone_no_m', 'label'], axis=1), df_train['label'],
                                                          test_size=0.2,
                                                          random_state=2020)

    params = {'objective': 'binary',
              'boosting': 'gbdt',
              'metric': 'auc',
              'learning_rate': 0.1,
              'num_leaves': 31,
              'lambda_l1': 0.1,
              'lambda_l2': 0,
              'min_data_in_leaf': 20,
              'is_unbalance': True,
              'max_depth': -1,
              'seed': 2020}

    train_cols = [i for i in df_train.columns if i not in ['phone_no_m', 'label']]
    useful_cols = {}
    useless_cols = {}

    for i in train_cols:
        print(i)

        lgb_train = lgb.Dataset(X_train[[i]].values, y_train)
        lgb_valid = lgb.Dataset(X_valid[[i]].values, y_valid, reference=lgb_train)
        lgb_test = lgb.train(params,
                             lgb_train,
                             num_boost_round=1000,
                             valid_sets=[lgb_valid, lgb_train],
                             early_stopping_rounds=50,
                             verbose_eval=20)

        print('*' * 5)
        print(lgb_test.best_score['valid_0']['auc'])
        if lgb_test.best_score['valid_0']['auc'] > auc_threshold:
            useful_cols[i] = lgb_test.best_score['valid_0']['auc']
        else:
            useless_cols[i] = lgb_test.best_score['valid_0']['auc']
        print('*' * 20)
        print('\n')


def correlation(df, useful_cols, threshold=0.98):
    """
    去除特征相关系数大于阈值的特征，保留AUC较大的特征
    :param df:
    :param threshold: 阈值
    :param useful_cols: 包含特征AUC的字典
    :return:
    """
    col_corr = set()
    corr_matrix = df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colName_i = corr_matrix.columns[i]
                colName_j = corr_matrix.columns[j]
                if useful_cols[colName_i] >= useful_cols[colName_j]:
                    col_corr.add(colName_j)
                else:
                    col_corr.add(colName_i)

    return col_corr


col = correlation(df, 0.98, useful_cols)
print('Correlated columns: ', col)


# 首先计算出特征重要性
df_importance['normalized_importance'] = df_importance['importance'] / df_importance['importance'].sum()
df_importance['cumulative_importance'] = np.cumsum(df_importance['normalized_importance'])
record_low_importance = df_importance[df_importance['cumulative_importance'] > 0.99]
to_drop = list(record_low_importance['feature_name'])
print(to_drop)
