# -*- coding: utf-8 -*-
# @Time     : 2020/10/19 11:24
# @Author   : Michael_Zhouy

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve, f1_score
import gc


def self_metric(preds, train_data):
    # 自定义评估函数
    labels = train_data.get_label()
    fpr, tpr, _ = roc_curve(labels, preds)
    max_tpr = tpr[np.where(fpr < 0.001)][-1]

    return 'self_metric', max_tpr, True


def self_metric(preds, train_data):
    # 自定义评估函数（特定阈值下的f1）
    labels = train_data.get_label()
    y_preds = np.where(preds >= np.percentile(preds, 95), 1, 0)
    f1 = f1_score(labels, y_preds)
    return 'self_metric', f1, True


def lgb_model(X_train, y_train, X_valid=None, y_valid=None, valid_model_path='./'):
    """
    lgb训练
    @param X_train:
    @param y_train:
    @param X_valid:
    @param y_valid:
    @return:
    """
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

    train_dataset = lgb.Dataset(X_train, label=y_train)
    if X_valid != None:
        valid_dataset = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train)
        valid_model = lgb.train(
            params,
            train_dataset,
            valid_sets=[valid_dataset, train_dataset],
            num_boost_round=1000000,
            early_stopping_rounds=200,
            verbose_eval=300
        )
        # 保存模型
        valid_model.save_model(valid_model_path + 'lgb.txt')
    else:
        # 加载模型
        valid_model = lgb.Booster(valid_model_path + 'lgb.txt')
        train_model = lgb.train(
            params,
            train_dataset,
            num_boost_round=valid_model.best_iteration+20
        )
        # 导出特征重要性
        importance = train_model.feature_importance(importance_type='gain')
        feature_name = train_model.feature_name()

        feature_importance = pd.DataFrame({
            'feature_name': feature_name,
            'importance': importance
        }).sort_values(by='importance', ascending=False)
        feature_importance.to_csv('imp.csv', index=False)


def auc_select(X_train, y_train, X_valid, y_valid, cols, threshold=0.52):
    """
    基于AUC的单特征筛选
    @param X_train:
    @param y_train:
    @param X_valid:
    @param y_valid:
    @param cols:
    @return:
    """
    useful_cols = []
    useless_cols = []
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
    for i in cols:
        print(i)
        lgb_train = lgb.Dataset(X_train[[i]].values, y_train)
        lgb_valid = lgb.Dataset(X_valid[[i]].values, y_valid, reference=lgb_train)
        lgb_test = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_valid, lgb_train],
            num_boost_round=1000,
            early_stopping_rounds=50,
            verbose_eval=20
        )
        print('*' * 10)
        print(lgb_test.best_score['valid_0']['auc'])
        if lgb_test.best_score['valid_0']['auc'] > threshold:
            useful_cols.append(i)
        else:
            useless_cols.append(i)
    return useful_cols, useless_cols


def correlation(df, useful_cols, threshold=0.98):
    """
    去除特征相关系数大于阈值的特征
    @param df:
    @param threshold:
    @param useful_cols:
    @return:
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


def lgb_5_folds(X, y, X_test, sub, save_path='./', oof=[], imp_list=[]):
    """
    5折交叉验证
    @param X:
    @param y:
    @param X_test:
    @param sub:
    @param save_path:
    @return:
    """
    prediction = pd.DataFrame()
    skf = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
    for fold_id, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
        print('\nFold_{} Training ==============\n'.format(fold_id + 1))
        X_train = X.iloc[trn_idx]
        Y_train = y.iloc[trn_idx]
        X_val = X.iloc[val_idx]
        Y_val = y.iloc[val_idx]

        lgb_train = lgb.Dataset(X_train, Y_train)
        lgb_valid = lgb.Dataset(X_val, Y_val, reference=lgb_train)

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
            'seed': fold_id
        }
        lgb_model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_valid, lgb_train],
            num_boost_round=1000000,
            early_stopping_rounds=200,
            verbose_eval=200
        )

        imp = lgb_model.feature_importance(importance_type='gain')
        feat_name = lgb_model.feature_name()
        df_imp = pd.DataFrame({
            'feature_name': feat_name,
            'importance': imp
        })
        imp_list.append(df_imp)

        pred_val = lgb_model.predict(X_val)
        # df_oof = df_train.iloc[val_idx][['id', ycol]].copy()
        # df_oof['pred'] = pred_val
        # oof.append(df_oof)

        f1_best = 0
        for i in np.arange(0.1, 1, 0.01):
            y_valid_pred = np.where(pred_val > i, 1, 0)
            f1 = np.round(f1_score(Y_val, y_valid_pred), 5)
            if f1 > f1_best:
                threshold = i
                f1_best = f1
        print('threshold: ', threshold)
        y_valid_pred = np.where(pred_val > threshold, 1, 0)
        print('Valid F1: ', np.round(f1_score(Y_val, y_valid_pred), 5))
        print('Valid mean label: ', np.mean(y_valid_pred))

        pred_test = np.where(lgb_model.predict(X_test) > threshold, 1, 0)
        prediction['label_{}'.format(fold_id)] = pred_test

        del lgb_model, pred_val, pred_test, X_train, Y_train, X_val, Y_val
        gc.collect()

    feat_imp = pd.concat(imp_list)
    feat_imp = feat_imp.groupby('feature_name')['importance'].agg('mean').sort_values(ascending=False).reset_index()
    feat_imp.to_csv(save_path + 'imp.csv', index=False)

    prediction['label_mean'] = prediction.apply('mean', axis=1)
    prediction['label'] = np.where(prediction['label_mean'] > 0.5, 1, 0)
    print('prediction: \n', prediction.head())

    print('Test mean label: ', np.mean(prediction['label']))
    sub['label'] = prediction['label'].values
    print('sub: \n', sub.head())
    sub.to_csv(save_path + 'sub.csv', index=False)
