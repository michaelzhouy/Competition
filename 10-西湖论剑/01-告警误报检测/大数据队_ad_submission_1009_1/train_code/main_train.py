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

def train_func(train_path, test_path):
    # 请填写训练代码
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    data = pd.concat([train, test])
    del train, test
    gc.collect()

    num_cols = ['bytesOut', 'bytesIn', 'txId']
    single_cols = ['transProtocol', 'appProtocol', 'name']
    null_cols = ['srcGeoCity', 'srcGeoAddress', 'srcGeoLatitude', 'srcGeoLongitude', 'destGeoAddress']
    notused_cols = ['srcAddress', 'srcPort', 'destAddress', 'destPort', 'destGeoLatitude', 'destGeoLongitude',
                    'requestUrlQuery', 'requestUrl', 'httpReferer', 'requestBody']
    drop_cols = single_cols + null_cols + notused_cols + ['startTime']
    data.drop(drop_cols, axis=1, inplace=True)
    gc.collect()

    cat_cols = ['destGeoCountry', 'destGeoRegion', 'destGeoCity', 'catOutcome', 'destHostName', 'responseCode']

    data['destAddress'] = data['destGeoCountry'] + data['destGeoRegion'] + data['destGeoCity']
    cat_cols.append('destAddress')

    upper_cols = ['requestMethod', 'httpVersion']
    for i in upper_cols:
        data[i] = data[i].str.upper()
        cat_cols.append(i)

    data['accessAgent_0'] = data['accessAgent'].astype(str).apply(lambda x: re.split('/| ', x)[0])
    cat_cols.append('accessAgent_0')

    data['responseCode_0'] = data['responseCode'].apply(lambda x: str(x)[0])
    cat_cols.append('responseCode_0')

    header_cols = ['requestHeader', 'responseHeader']
    for i in header_cols:
        data[i + '_0'] = data[i].apply(lambda x: str(x).split(':')[0])
        cat_cols.append(i + '_0')

    content_cols = ['requestContentType', 'responseContentType']
    for i in content_cols:
        data[i + '_0'] = data[i].apply(lambda x: str(x).split('/')[0])
        cat_cols.append(i + '_0')

    for i in cat_cols:
        lbl = LabelEncoder()
        data[i] = lbl.fit_transform(data[i].astype(str))
        data[i] = data[i].astype('category')

    used_cols = num_cols + cat_cols

    train = data.loc[data['label'].notnull(), :]
    test = data.loc[data['label'].isnull(), :]
    sub = test[['eventId']]

    y = train['label']
    X = train[used_cols]
    X_test = test[used_cols]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=2020)

    train_dataset = lgb.Dataset(X_train, y_train)
    valid_dataset = lgb.Dataset(X_valid, y_valid, reference=train_dataset)
    all_dataset = lgb.Dataset(X, y, reference=train_dataset)

    params = {'objective': 'binary',
              'boosting': 'gbdt',
              'metric': 'auc',
              # 'metric': 'None',  # 用自定义评估函数是将metric设置为'None'
              'num_iterations': 1000000,
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
    y_valid_pred = np.where(valid_model.predict(X_valid) > 0.5, 1, 0)
    print('Valid F1: ', f1_score(y_valid, y_valid_pred))

    # train_model = lgb.train(params,
    #                         all_dataset,
    #                         num_boost_round=valid_model.best_iteration+100)
    # y_test_pred = np.where(valid_model.predict(X_valid) > 0.5, 1, 0)


if __name__ == '__main__':
    train_path = '../data/train.csv'
    test_path = '../data/test_1.csv'
    train_func(train_path, test_path)
