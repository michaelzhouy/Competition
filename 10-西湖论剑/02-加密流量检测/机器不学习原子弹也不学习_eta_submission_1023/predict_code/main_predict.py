# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb


def count_encode(df, cols=[]):
    """
    count编码
    @param df:
    @param cols:
    @return:
    """
    for col in cols:
        print(col)
        vc = df[col].value_counts(dropna=True, normalize=True)
        df[col + '_count'] = df[col].map(vc).astype('float32')
    return df


def cross_cat_num(df, cat_col, num_col):
    """
    类别特征与数据特征groupby统计
    @param df:
    @param cat_col: 类别特征
    @param num_col: 数值特征
    @return:
    """
    def max_min(s):
        return s.max() - s.min()
    def quantile(s, q=0.25):
        return s.quantile(q)
    for f1 in cat_col:
        g = df.groupby(f1, as_index=False)
        for f2 in num_col:
            tmp = g[f2].agg({
                '{}_{}_count'.format(f1, f2): 'count',
                '{}_{}_max'.format(f1, f2): 'max',
                '{}_{}_min'.format(f1, f2): 'min',
                '{}_{}_median'.format(f1, f2): 'median',
                '{}_{}_mean'.format(f1, f2): 'mean',
                '{}_{}_sum'.format(f1, f2): 'sum',
                '{}_{}_skew'.format(f1, f2): 'skew',
                '{}_{}_std'.format(f1, f2): 'std',
                '{}_{}_nunique'.format(f1, f2): 'nunique',
                '{}_{}_max_min'.format(f1, f2): max_min,
                '{}_{}_quantile_25'.format(f1, f2): lambda x: quantile(x, 0.25),
                '{}_{}_quantile_75'.format(f1, f2): lambda x: quantile(x, 0.75)
            })
            df = df.merge(tmp, on=f1, how='left')
    return df


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


def test_func(test_path,save_path):
    # 请填写测试代码
    test = pd.read_csv(test_path)
    submission = test[['eventId']]
    # 选手不得改变格式，测试代码跑不通分数以零算

    single_cols = ['appProtocol']
    test.drop(single_cols, axis=1, inplace=True)

    cat_cols = ['srcAddress', 'destAddress',
                'tlsVersion', 'tlsSubject', 'tlsIssuerDn', 'tlsSni']

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
    test['tlsVersion_map'] = test['tlsVersion'].map(tlsVersion_map)
    cat_cols.append('tlsVersion_map')

    cat_cols += ['srcAddressPort', 'destAddressPort']
    num_cols = ['bytesOut', 'bytesIn', 'pktsIn', 'pktsOut']

    test = count_encode(test, cat_cols)
    test = cross_cat_num(test, cat_cols, num_cols)
    test = arithmetic(test, num_cols)

    used_cols = [i for i in test.columns if i not in ['eventId']]
    test = test[used_cols].copy()

    psi_drop_cols = ['tlsSubject', 'destAddress', 'srcAddress', 'srcAddressPort', 'tlsIssuerDn', 'tlsSni',
                     'tlsVersion_map', 'destAddressPort', 'tlsVersion']
    test.drop(psi_drop_cols, axis=1, inplace=True)

    print('test.shape: \n', test.shape)
    used_cols = ['srcPort', 'destPort', 's2d_nunique', 'd2sp_nunique', 'srcAddress_bytesOut_min', 'srcAddress_bytesOut_median', 'srcAddress_bytesOut_mean', 'srcAddress_bytesOut_skew', 'srcAddress_bytesOut_nunique', 'srcAddress_bytesOut_max_min', 'srcAddress_bytesOut_quantile_25', 'srcAddress_bytesOut_quantile_75', 'srcAddress_bytesIn_min', 'srcAddress_bytesIn_skew', 'srcAddress_bytesIn_std', 'srcAddress_bytesIn_nunique', 'srcAddress_bytesIn_quantile_25', 'srcAddress_pktsIn_mean', 'srcAddress_pktsIn_sum', 'srcAddress_pktsIn_skew', 'srcAddress_pktsIn_nunique', 'srcAddress_pktsIn_quantile_25', 'srcAddress_pktsOut_mean', 'srcAddress_pktsOut_sum', 'srcAddress_pktsOut_skew', 'srcAddress_pktsOut_std', 'srcAddress_pktsOut_nunique', 'srcAddress_pktsOut_quantile_75', 'destAddress_bytesOut_min', 'destAddress_bytesOut_median', 'destAddress_bytesOut_skew', 'destAddress_bytesOut_nunique', 'destAddress_bytesOut_quantile_25', 'destAddress_bytesIn_median', 'destAddress_bytesIn_skew', 'destAddress_pktsIn_sum', 'destAddress_pktsIn_skew', 'destAddress_pktsIn_std', 'destAddress_pktsIn_nunique', 'destAddress_pktsOut_min', 'destAddress_pktsOut_median', 'destAddress_pktsOut_mean', 'destAddress_pktsOut_skew', 'destAddress_pktsOut_std', 'destAddress_pktsOut_nunique', 'destAddress_pktsOut_quantile_25', 'tlsVersion_bytesOut_quantile_25', 'tlsVersion_bytesIn_quantile_25', 'tlsVersion_bytesIn_quantile_75', 'tlsVersion_pktsIn_min', 'tlsVersion_pktsIn_mean', 'tlsVersion_pktsIn_quantile_75', 'tlsVersion_pktsOut_min', 'tlsVersion_pktsOut_median', 'tlsVersion_pktsOut_skew', 'tlsVersion_pktsOut_std', 'tlsVersion_pktsOut_nunique', 'tlsVersion_pktsOut_max_min', 'tlsVersion_pktsOut_quantile_25', 'tlsVersion_pktsOut_quantile_75', 'tlsSubject_bytesOut_min', 'tlsSubject_bytesOut_median', 'tlsSubject_bytesOut_skew', 'tlsSubject_bytesOut_std', 'tlsSubject_bytesOut_nunique', 'tlsSubject_bytesOut_max_min', 'tlsSubject_bytesOut_quantile_25', 'tlsSubject_bytesOut_quantile_75', 'tlsSubject_bytesIn_min', 'tlsSubject_bytesIn_sum', 'tlsSubject_bytesIn_skew', 'tlsSubject_bytesIn_nunique', 'tlsSubject_bytesIn_max_min', 'tlsSubject_pktsIn_mean', 'tlsSubject_pktsIn_sum', 'tlsSubject_pktsIn_skew', 'tlsSubject_pktsIn_nunique', 'tlsSubject_pktsIn_max_min', 'tlsSubject_pktsIn_quantile_25', 'tlsSubject_pktsIn_quantile_75', 'tlsSubject_pktsOut_count', 'tlsSubject_pktsOut_median', 'tlsSubject_pktsOut_mean', 'tlsSubject_pktsOut_sum', 'tlsSubject_pktsOut_skew', 'tlsSubject_pktsOut_std', 'tlsSubject_pktsOut_nunique', 'tlsSubject_pktsOut_quantile_25', 'tlsIssuerDn_bytesOut_min', 'tlsIssuerDn_bytesOut_skew', 'tlsIssuerDn_bytesOut_std', 'tlsIssuerDn_bytesOut_quantile_25', 'tlsIssuerDn_bytesIn_min', 'tlsIssuerDn_bytesIn_median', 'tlsIssuerDn_bytesIn_mean', 'tlsIssuerDn_bytesIn_sum', 'tlsIssuerDn_bytesIn_skew', 'tlsIssuerDn_bytesIn_max_min', 'tlsIssuerDn_bytesIn_quantile_75', 'tlsIssuerDn_pktsIn_min', 'tlsIssuerDn_pktsIn_mean', 'tlsIssuerDn_pktsIn_sum', 'tlsIssuerDn_pktsIn_skew', 'tlsIssuerDn_pktsIn_std', 'tlsIssuerDn_pktsIn_nunique', 'tlsIssuerDn_pktsIn_max_min', 'tlsIssuerDn_pktsOut_count', 'tlsIssuerDn_pktsOut_sum', 'tlsIssuerDn_pktsOut_std', 'tlsIssuerDn_pktsOut_nunique', 'tlsIssuerDn_pktsOut_max_min', 'tlsSni_bytesOut_max', 'tlsSni_bytesOut_min', 'tlsSni_bytesOut_median', 'tlsSni_bytesOut_mean', 'tlsSni_bytesOut_sum', 'tlsSni_bytesOut_skew', 'tlsSni_bytesOut_std', 'tlsSni_bytesOut_nunique', 'tlsSni_bytesOut_max_min', 'tlsSni_bytesOut_quantile_25', 'tlsSni_bytesOut_quantile_75', 'tlsSni_bytesIn_min', 'tlsSni_bytesIn_sum', 'tlsSni_bytesIn_skew', 'tlsSni_bytesIn_nunique', 'tlsSni_pktsIn_min', 'tlsSni_pktsIn_median', 'tlsSni_pktsIn_sum', 'tlsSni_pktsIn_skew', 'tlsSni_pktsIn_nunique', 'tlsSni_pktsIn_max_min', 'tlsSni_pktsIn_quantile_25', 'tlsSni_pktsOut_count', 'tlsSni_pktsOut_median', 'tlsSni_pktsOut_mean', 'tlsSni_pktsOut_skew', 'tlsSni_pktsOut_std', 'tlsSni_pktsOut_nunique', 'tlsSni_pktsOut_quantile_75', 'srcAddressPort_bytesOut_std', 'srcAddressPort_bytesIn_std', 'srcAddressPort_pktsIn_sum', 'srcAddressPort_pktsIn_std', 'srcAddressPort_pktsOut_count', 'srcAddressPort_pktsOut_sum', 'srcAddressPort_pktsOut_std', 'destAddressPort_bytesOut_quantile_75', 'destAddressPort_bytesIn_min', 'destAddressPort_bytesIn_nunique', 'destAddressPort_bytesIn_quantile_25', 'destAddressPort_pktsIn_min', 'destAddressPort_pktsOut_sum', 'bytesOut_bytesIn_add', 'bytesOut_bytesIn_subtract', 'bytesOut_bytesInc_multiply', 'bytesOut_pktsIn_add', 'bytesOut_pktsInc_multiply', 'bytesOut_pktsOutc_multiply', 'bytesIn_pktsOut_subtract', 'pktsIn_pktsOut_add', 'pktsIn_pktsOut_subtract', 'bytesOut_bytesIn_ratio', 'bytesOut_pktsIn_ratio', 'bytesOut_pktsOut_ratio', 'bytesIn_bytesOut_ratio', 'bytesIn_pktsIn_ratio', 'bytesIn_pktsOut_ratio', 'pktsIn_bytesOut_ratio', 'pktsIn_bytesIn_ratio', 'pktsIn_pktsOut_ratio', 'pktsOut_bytesOut_ratio', 'pktsOut_bytesIn_ratio', 'pktsOut_pktsIn_ratio']
    X_test = test[used_cols].copy()
    print('X_test.shape: \n', X_test.shape)

    train_model = lgb.Booster(model_file='../train_code/lgb.txt')
    train_model.save_model('./lgb.txt')

    pred = train_model.predict(X_test)
    threshold = np.load('../train_code/threshold.npy')
    y_pred = np.where(pred >= threshold, 1, 0)

    submission['label'] = y_pred
    print('y_pred.mean(): \n', y_pred.mean())

    submission.to_csv(save_path + '机器不学习原子弹也不学习_eta_submission_1023.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    test_path = '../data/test_1.csv'
    sava_path = '../result/'
    test_func(test_path, sava_path)
