# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
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


def test_func(test_path,save_path):
    # 请填写测试代码
    test = pd.read_csv(test_path)
    submission = test[['eventId']]
    # 选手不得改变格式，测试代码跑不通分数以零算

    single_cols = ['appProtocol']
    test.drop(single_cols, axis=1, inplace=True)

    cat_cols = ['srcAddress', 'destAddress', 'tlsVersion',
                'tlsSubject', 'tlsIssuerDn', 'tlsSni']

    test['srcAddressPort'] = test['srcAddress'].astype(str) + test['srcPort'].astype(str)
    test['destAddressPort'] = test['destAddress'].astype(str) + test['destPort'].astype(str)

    # srcAddress To destAddress
    tmp = test.groupby('srcAddress', as_index=False)['destAddress'].agg({
        's2d_count': 'count',
        's2d_nunique': 'nunique'
    })
    test = test.merge(tmp, on='srcAddress', how='left')

    # srcAddressPort To destAddressPort
    tmp = test.groupby('srcAddressPort', as_index=False)['destAddressPort'].agg({
        'sp2dp_count': 'count',
        'sp2dp_nunique': 'nunique'
    })
    test = test.merge(tmp, on='srcAddressPort', how='left')

    # srcAddress To destAddressPort
    tmp = test.groupby('srcAddress', as_index=False)['destAddressPort'].agg({
        's2dp_count': 'count',
        's2dp_nunique': 'nunique'
    })
    test = test.merge(tmp, on='srcAddress', how='left')

    # srcAddressPort To destAddress
    tmp = test.groupby('srcAddressPort', as_index=False)['destAddress'].agg({
        'sp2d_count': 'count',
        'sp2d_nunique': 'nunique'
    })
    test = test.merge(tmp, on='srcAddressPort', how='left')

    # destAddress To srcAddress
    tmp = test.groupby('destAddress', as_index=False)['srcAddress'].agg({
        'd2s_count': 'count',
        'd2s_nunique': 'nunique'
    })
    test = test.merge(tmp, on='destAddress', how='left')

    # destAddressPort To srcAddressPort
    tmp = test.groupby('destAddressPort', as_index=False)['srcAddressPort'].agg({
        'dp2sp_count': 'count',
        'dp2sp_nunique': 'nunique'
    })
    test = test.merge(tmp, on='destAddressPort', how='left')

    # destAddressPort To srcAddress
    tmp = test.groupby('destAddressPort', as_index=False)['srcAddress'].agg({
        'dp2s_count': 'count',
        'dp2s_nunique': 'nunique'
    })
    test = test.merge(tmp, on='destAddressPort', how='left')

    # destAddress To srcAddressProt
    tmp = test.groupby('destAddress', as_index=False)['srcAddressPort'].agg({
        'd2sp_count': 'count',
        'd2sp_nunique': 'nunique'
    })
    test = test.merge(tmp, on='destAddress', how='left')

    cat_cols += ['srcAddressPort', 'destAddressPort']
    num_cols = ['bytesOut', 'bytesIn', 'pktsIn', 'pktsOut']

    arithmetic(test, num_cols)

    count_encode(test, cat_cols)
    test = cross_cat_num(test, cat_cols, num_cols)

    test.drop(cat_cols, axis=1, inplace=True)

    used_cols = [i for i in test.columns if i not in ['eventId', 'label']]

    test = test[used_cols]

    col_corr = ['tlsIssuerDn_pktsOut_max', 'srcAddressPort_pktsIn_mean', 'destAddressPort_pktsOut_max', 'destAddressPort_pktsIn_mean', 'destAddressPort_pktsOut_nunique', 'd2sp_nunique', 'srcAddress_pktsOut_min', 'tlsVersion_bytesIn_sum', 'destAddressPort_pktsOut_mean', 'srcAddressPort_bytesOut_max', 'tlsSubject_pktsOut_sum', 'destAddressPort_bytesOut_min', 'tlsIssuerDn_bytesOut_nunique', 'srcAddressPort_bytesOut_median', 'bytesIn_pktsIn_subtract', 'srcAddressPort_bytesIn_min', 'tlsIssuerDn_bytesOut_mean', 'tlsSni_bytesIn_max', 'tlsVersion_bytesIn_median', 'tlsVersion_pktsIn_sum', 'tlsVersion_pktsIn_nunique', 'srcAddressPort_pktsOut_min', 'tlsVersion_bytesOut_sum', 'tlsIssuerDn_pktsOut_min', 'destAddressPort_pktsIn_median', 'srcAddressPort_bytesOut_sum', 'tlsIssuerDn_bytesIn_nunique', 'destAddress_count', 'srcAddressPort_pktsOut_median', 'tlsSubject_pktsOut_max', 'tlsIssuerDn_pktsOut_median', 'tlsIssuerDn_pktsOut_mean', 'destAddressPort_bytesIn_median', 'sp2d_count', 'destAddress_pktsIn_mean', 'destAddress_bytesIn_mean', 'tlsVersion_bytesIn_nunique', 'srcAddressPort_count', 'tlsIssuerDn_pktsOut_skew', 'bytesIn_pktsOutc_multiply', 'destAddressPort_bytesOut_max', 'tlsVersion_pktsOut_sum', 'tlsVersion_bytesOut_nunique', 'tlsVersion_pktsOut_max', 'destAddressPort_pktsIn_skew', 'bytesOut_pktsIn_add', 'tlsVersion_pktsOut_nunique', 'tlsSni_pktsIn_mean', 'srcAddressPort_pktsIn_median', 'destAddressPort_bytesIn_min', 'bytesOut_pktsOut_add', 'tlsVersion_bytesIn_mean', 'tlsSni_bytesIn_median', 'd2s_count', 'srcAddress_bytesIn_sum', 'bytesOut_pktsIn_subtract', 'srcAddressPort_pktsIn_sum', 'tlsIssuerDn_pktsOut_sum', 'srcAddressPort_bytesIn_median', 'srcAddressPort_pktsIn_max', 'srcAddress_pktsOut_sum', 'destAddressPort_bytesIn_nunique', 'bytesIn_pktsOut_subtract', 's2dp_count', 'destAddressPort_pktsIn_max', 'destAddressPort_pktsOut_sum', 'destAddress_pktsOut_sum', 'dp2s_count', 'srcAddressPort_bytesIn_sum', 'srcAddressPort_bytesIn_max', 'destAddressPort_pktsOut_min', 's2dp_nunique', 'pktsIn_pktsOutc_multiply', 'bytesIn_pktsIn_add', 'destAddressPort_bytesIn_mean', 'srcAddress_pktsOut_median', 'bytesIn_pktsInc_multiply', 'srcAddressPort_pktsOut_max', 'tlsVersion_pktsOut_skew', 'tlsVersion_pktsIn_skew', 'dp2sp_count', 'srcAddressPort_pktsIn_min', 'destAddressPort_bytesOut_mean', 'tlsSubject_bytesIn_mean', 'srcAddressPort_pktsOut_sum', 'tlsVersion_bytesIn_skew', 'tlsVersion_pktsOut_mean', 'd2sp_count', 'tlsSubject_pktsOut_min', 'tlsSni_pktsOut_sum', 'srcAddressPort_bytesOut_mean', 'destAddressPort_pktsIn_nunique', 'destAddressPort_bytesOut_nunique', 'destAddressPort_bytesIn_sum', 'srcAddressPort_bytesOut_min', 'destAddress_bytesIn_sum', 'destAddressPort_bytesIn_max', 'destAddressPort_pktsOut_skew', 'tlsSni_pktsOut_min', 'tlsVersion_pktsIn_max', 'srcAddress_count', 'destAddressPort_count', 'srcAddressPort_bytesIn_mean', 'tlsVersion_pktsOut_median', 'destAddressPort_bytesOut_sum', 'destAddressPort_pktsOut_median', 'destAddressPort_pktsIn_min', 'bytesIn_pktsOut_add', 'destAddressPort_pktsIn_sum', 'destAddressPort_bytesOut_median', 'destAddressPort_bytesIn_skew', 'tlsSni_pktsOut_max', 'tlsSni_bytesIn_mean', 'destAddressPort_bytesOut_skew', 'bytesOut_pktsOut_subtract', 'dp2sp_nunique', 'srcAddressPort_pktsOut_mean', 'tlsVersion_bytesIn_max']

    test.drop(list(col_corr), axis=1, inplace=True)

    thresholds = [0.7699999999999997, 0.1, 0.44999999999999984, 0.8199999999999996, 0.1]
    lgb_0 = lgb.Booster(model_file='../train_code/lgb_0.txt')
    lgb_1 = lgb.Booster(model_file='../train_code/lgb_1.txt')
    lgb_2 = lgb.Booster(model_file='../train_code/lgb_2.txt')
    lgb_3 = lgb.Booster(model_file='../train_code/lgb_3.txt')
    lgb_4 = lgb.Booster(model_file='../train_code/lgb_4.txt')

    lgb_0.save_model('./lgb_0.txt')
    lgb_1.save_model('./lgb_1.txt')
    lgb_2.save_model('./lgb_2.txt')
    lgb_3.save_model('./lgb_3.txt')
    lgb_4.save_model('./lgb_4.txt')

    prediction = pd.DataFrame()
    prediction['pred0'] = np.where(lgb_0.predict(test) > thresholds[0], 1, 0)
    prediction['pred1'] = np.where(lgb_1.predict(test) > thresholds[1], 1, 0)
    prediction['pred2'] = np.where(lgb_2.predict(test) > thresholds[2], 1, 0)
    prediction['pred3'] = np.where(lgb_3.predict(test) > thresholds[3], 1, 0)
    prediction['pred4'] = np.where(lgb_4.predict(test) > thresholds[4], 1, 0)

    prediction['label_sum'] = prediction.apply('sum', axis=1)
    prediction['label'] = np.where(prediction['label_sum'] > 2, 1, 0)
    print('Test mean label: ', np.mean(prediction['label']))
    submission['label'] = prediction['label'].values

    submission.to_csv(save_path + '机器不学习原子弹也不学习_eta_submission_1016.csv',index = False,encoding='utf-8')


if __name__ == '__main__':
    test_path = '../data/test_1.csv'
    sava_path = '../result/'
    test_func(test_path, sava_path)
