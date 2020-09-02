# -*- coding:utf-8 -*-
# Time   : 2020/4/11 19:51
# Email  : 15602409303@163.com
# Author : Zhou Yang

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
import gc
import category_encoders as ce


def mean_woe_target_encoder(train, test, target, col, n_splits=10):
    folds = StratifiedKFold(n_splits)

    y_oof = np.zeros(train.shape[0])
    y_oof_2 = np.zeros(train.shape[0])
    y_test_oof = np.zeros(test.shape[0]).reshape(-1, 1)
    y_test_oof2 = np.zeros(test.shape[0]).reshape(-1, 1)

    splits = folds.split(train, target)

    for fold_n, (train_index, valid_index) in enumerate(splits):
        X_train, X_valid = train[col].iloc[train_index], train[col].iloc[valid_index]
        y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
        clf = ce.target_encoder.TargetEncoder()

        #    dtrain = lgb.Dataset(X_train, label=y_train)
        #    dvalid = lgb.Dataset(X_valid, label=y_valid)

        # clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=1, early_stopping_rounds=500)
        clf.fit(X_train.values, y_train.values)
        y_pred_valid = clf.transform(X_valid.values)
        y_oof[valid_index] = y_pred_valid.values.reshape(1, -1)

        tp = (clf.transform(test[col].values) / (n_splits * 1.0)).values
        tp = tp.reshape(-1, 1)
        y_test_oof += tp

        del X_train, X_valid, y_train, y_valid
        gc.collect()

    for fold_n, (train_index, valid_index) in enumerate(splits):
        X_train, X_valid = train[col].iloc[train_index], train[col].iloc[valid_index]
        y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
        clf = ce.woe.WOEEncoder()

        #    dtrain = lgb.Dataset(X_train, label=y_train)
        #    dvalid = lgb.Dataset(X_valid, label=y_valid)

        # clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=1, early_stopping_rounds=500)
        clf.fit(X_train.values, y_train.values)
        y_pred_valid = clf.transform(X_valid.values)
        y_oof2[valid_index] = y_pred_valid.values.reshape(1, -1)

        tp = (clf.transform(test[col].values) / (n_splits * 1.0)).values
        tp = tp.reshape(-1, 1)
        y_test_oof2 += tp
        del X_train, X_valid, y_train, y_valid
        gc.collect()
    return y_oof, y_oof_2, y_test_oof, y_test_oof2