# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


def count_encode(df, cols=[]):
    for col in cols:
        print(col)
        vc = df[col].value_counts(dropna=True, normalize=True)
        df[col + '_count'] = df[col].map(vc).astype('float32')


def train_func(train_path, test_path, save_path):
    # 请填写训练代码
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    data = pd.concat([train, test])
    del train, test
    gc.collect()

    single_cols = ['appProtocol']
    data.drop(single_cols, axis=1, inplace=True)
    gc.collect()

    cat_cols = ['srcAddress', 'destAddress', 'tlsVersion',
                'tlsSubject', 'tlsIssuerDn', 'tlsSni']

    data['srcAddressPort'] = data['srcAddress'].astype(str) + data['srcPort'].astype(str)
    data['destAddressPort'] = data['destAddress'].astype(str) + data['destPort'].astype(str)
    cat_cols += ['srcAddressPort', 'destAddressPort']
    num_cols = ['bytesOut', 'bytesIn', 'pktsIn', 'pktsOut']
    data['bytesOut-bytesIn'] = data['bytesOut'] - data['bytesIn']
    data['pktsOut-pktsIn'] = data['pktsOut'] - data['pktsIn']

    tlsVersion_map = {
        'TLSv1': 1,
        'TLS 1.2': 1,
        'TLS 1.3': 1,
        'SSLv2': 2,
        'SSLv3': 3,
        '0x4854': 4,
        '0x4752': 4,
        'UNDETERMINED': 5
    }
    data['tlsVersion_map'] = data['tlsVersion'].map(tlsVersion_map)
    cat_cols.append('tlsVersion_map')

    count_encode(data, cat_cols)

    for i in cat_cols:
        lbl = LabelEncoder()
        data[i] = lbl.fit_transform(data[i].astype(str))
        data[i] = data[i].astype('category')

    used_cols = [i for i in data.columns if i not in ['eventId', 'label']]
    train = data.loc[data['label'].notnull(), :]
    test = data.loc[data['label'].isnull(), :]
    sub = test[['eventId']]

    y = train['label']
    X = train[used_cols]
    X_test = test[used_cols]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=2020)

    train_dataset = lgb.Dataset(X_train, y_train)
    valid_dataset = lgb.Dataset(X_valid, y_valid, reference=train_dataset)
    all_dataset = lgb.Dataset(train[used_cols], train['label'], reference=train_dataset)

    params = {'objective': 'binary',
              'boosting': 'gbdt',
              'metric': 'auc',
              # 'metric': 'None',  # 用自定义评估函数是将metric设置为'None'
              'num_boost_round': 1000000,
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
                            verbose_eval=300)

    pred = valid_model.predict(X_valid)

    f1_best = 0
    for i in np.arange(0.1, 1, 0.01):
        y_valid_pred = np.where(pred > i, 1, 0)
        f1 = np.round(f1_score(y_valid, y_valid_pred), 5)
        #         print('f1: ', f1)
        if f1 > f1_best:
            threshold = i
            f1_best = f1

    print('threshold: ', threshold)
    y_valid_pred = np.where(pred > threshold, 1, 0)
    print('Valid F1: ', np.round(f1_score(y_valid, y_valid_pred), 5))
    print('Valid mean label: ', np.mean(y_valid_pred))

    params = {'objective': 'binary',
              'boosting': 'gbdt',
              'metric': 'auc',
              # 'metric': 'None',  # 用自定义评估函数是将metric设置为'None'
              # 'num_boost_round': 1000000,
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
    train_model = lgb.train(params,
                            all_dataset,
                            num_boost_round=valid_model.best_iteration+20)
    y_test_pred = np.where(train_model.predict(X_test) > threshold, 1, 0)

    print('Test mean label: ', np.mean(y_test_pred))
    sub['label'] = y_test_pred
    sub.to_csv(save_path + '机器不学习原子弹也不学习_eta_submission_1014.csv', index=False)


if __name__ == '__main__':
    train_path = '../data/train.csv'
    test_path = '../data/test_1.csv'
    save_path = '../result/'
    train_func(train_path, test_path, save_path)
