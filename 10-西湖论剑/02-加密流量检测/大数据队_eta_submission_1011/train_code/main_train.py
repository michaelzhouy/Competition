# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb


def arithmetic(df, cross_features):
    """
    数值特征之间的加减乘除
    @param df:
    @param cross_features: 交叉用的数值特征
    @return:
    """
    for i in range(len(cross_features)):
        for j in range(i + 1, len(cross_features)):
            colname_add = '{}_{}_add'.format(cross_features[i], cross_features[j])
            colname_substract = '{}_{}_subtract'.format(cross_features[i], cross_features[j])
            colname_multiply = '{}_{}c_multiply'.format(cross_features[i], cross_features[j])
            df[colname_add] = df[cross_features[i]] + df[cross_features[j]]
            df[colname_substract] = df[cross_features[i]] - df[cross_features[j]]
            df[colname_multiply] = df[cross_features[i]] * df[cross_features[j]]

    for f1 in cross_features:
        for f2 in cross_features:
            if f1 != f2:
                colname_ratio = '{}_{}_ratio'.format(f1, f2)
                df[colname_ratio] = df[f1].values / (df[f2].values + 0.001)
    return df


def correlation(df, threshold=0.98):
    """
    特征相关性计算
    @param df:
    @param threshold:
    @return:
    """
    col_corr = set()
    corr_matrix = df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colName = corr_matrix.columns[i]
                col_corr.add(colName)
    return col_corr


def count_encode(df, cols=[]):
    for col in cols:
        print(col)
        vc = df[col].value_counts(dropna=True, normalize=True)
        df[col + '_count'] = df[col].map(vc).astype('float32')


# 交叉特征
def cross_cat_num(df, cat_col, num_col):
    for f1 in cat_col:
        g = df.groupby(f1, as_index=False)
        for f2 in num_col:
            df_new = g[f2].agg({
                '{}_{}_max'.format(f1, f2): 'max',
                '{}_{}_min'.format(f1, f2): 'min',
                '{}_{}_median'.format(f1, f2): 'median',
                '{}_{}_mean'.format(f1, f2): 'mean',
                '{}_{}_sum'.format(f1, f2): 'sum',
                '{}_{}_skew'.format(f1, f2): 'skew',
                '{}_{}_nunique'.format(f1, f2): 'nunique'
            })
            df = df.merge(df_new, on=f1, how='left')
    return df


def train_func(train_path, save_path):
    # 请填写训练代码
    train = pd.read_csv(train_path)

    single_cols = ['appProtocol']
    train.drop(single_cols, axis=1, inplace=True)

    cat_cols = ['srcAddress', 'destAddress', 'tlsVersion',
                'tlsSubject', 'tlsIssuerDn', 'tlsSni']

    train['srcAddressPort'] = train['srcAddress'].astype(str) + train['srcPort'].astype(str)
    train['destAddressPort'] = train['destAddress'].astype(str) + train['destPort'].astype(str)

    # srcAddress To destAddress
    tmp = train.groupby('srcAddress', as_index=False)['destAddress'].agg({
        's2d_count': 'count',
        's2d_nunique': 'nunique'
    })
    train = train.merge(tmp, on='srcAddress', how='left')

    # srcAddressPort To destAddressPort
    tmp = train.groupby('srcAddressPort', as_index=False)['destAddressPort'].agg({
        'sp2dp_count': 'count',
        'sp2dp_nunique': 'nunique'
    })
    train = train.merge(tmp, on='srcAddressPort', how='left')

    # srcAddress To destAddressPort
    tmp = train.groupby('srcAddress', as_index=False)['destAddressPort'].agg({
        's2dp_count': 'count',
        's2dp_nunique': 'nunique'
    })
    train = train.merge(tmp, on='srcAddress', how='left')

    # srcAddressPort To destAddress
    tmp = train.groupby('srcAddressPort', as_index=False)['destAddress'].agg({
        'sp2d_count': 'count',
        'sp2d_nunique': 'nunique'
    })
    train = train.merge(tmp, on='srcAddressPort', how='left')

    # destAddress To srcAddress
    tmp = train.groupby('destAddress', as_index=False)['srcAddress'].agg({
        'd2s_count': 'count',
        'd2s_nunique': 'nunique'
    })
    train = train.merge(tmp, on='destAddress', how='left')

    # destAddressPort To srcAddressPort
    tmp = train.groupby('destAddressPort', as_index=False)['srcAddressPort'].agg({
        'dp2sp_count': 'count',
        'dp2sp_nunique': 'nunique'
    })
    train = train.merge(tmp, on='destAddressPort', how='left')

    # destAddressPort To srcAddress
    tmp = train.groupby('destAddressPort', as_index=False)['srcAddress'].agg({
        'dp2s_count': 'count',
        'dp2s_nunique': 'nunique'
    })
    train = train.merge(tmp, on='destAddressPort', how='left')

    # destAddress To srcAddressProt
    tmp = train.groupby('destAddress', as_index=False)['srcAddressPort'].agg({
        'd2sp_count': 'count',
        'd2sp_nunique': 'nunique'
    })
    train = train.merge(tmp, on='destAddress', how='left')

    cat_cols += ['srcAddressPort', 'destAddressPort']
    num_cols = ['bytesOut', 'bytesIn', 'pktsIn', 'pktsOut']

    arithmetic(train, num_cols)

    count_encode(train, cat_cols)
    train = cross_cat_num(train, cat_cols, num_cols)

    train.drop(cat_cols, axis=1, inplace=True)

    used_cols = [i for i in train.columns if i not in ['eventId', 'label']]
    y = train['label']

    train = train[used_cols]

    col_corr = correlation(train, 0.98)
    print(col_corr)
    train.drop(list(col_corr), axis=1, inplace=True)

    X = train.copy()

    kfold = StratifiedKFold(n_splits=5)
    thresholds = []
    for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(X, y)):
        print('\nFold_{} Training ================================\n'.format(fold_id + 1))

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
        lgb_model = lgb.train(params,
                              lgb_train,
                              num_boost_round=10000,
                              valid_sets=[lgb_valid, lgb_train],
                              early_stopping_rounds=200,
                              verbose_eval=200)

        pred_val = lgb_model.predict(X_val)
        lgb_model.save_model('lgb_{}.txt'.format(fold_id))

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
        thresholds.append(threshold)
        print(thresholds)


if __name__ == '__main__':
    train_path = '../data/train.csv'
    save_path = '../result/'
    train_func(train_path, save_path)
