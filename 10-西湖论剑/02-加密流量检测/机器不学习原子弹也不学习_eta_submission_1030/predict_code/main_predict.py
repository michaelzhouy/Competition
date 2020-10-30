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


def test_func(test_path, save_path):
    # 请填写测试代码
    train = pd.read_csv(test_path)
    submission = train[['eventId']]

    single_cols = ['appProtocol']
    train.drop(single_cols, axis=1, inplace=True)

    cat_cols = ['srcAddress', 'destAddress',
                'tlsVersion', 'tlsSubject', 'tlsIssuerDn', 'tlsSni']

    train['srcAddressPort'] = train['srcAddress'].astype(str) + train['srcPort'].astype(str)
    train['destAddressPort'] = train['destAddress'].astype(str) + train['destPort'].astype(str)

    str_cols = ['srcAddress', 'destAddress', 'srcAddressPort', 'destAddressPort']
    count_nunique = {}
    for i in str_cols:
        for j in str_cols:
            if j == i:
                continue
            tr = train[i].groupby(train[j]).agg(['count', 'nunique'])
            train['{}_gp_{}_count'.format(i, j)] = train[j].map(tr['count'])
            train['{}_gp_{}_nunique'.format(i, j)] = train[j].map(tr['nunique'])
            count_nunique['{}_gp_{}'.format(i, j)] = tr
            train['{}_gp_{}_nunique_rate'.format(i, j)] = (train['{}_gp_{}_nunique'.format(i, j)]
                                                           / train['{}_gp_{}_count'.format(i, j)])
            train.drop(['{}_gp_{}_count'.format(i, j), '{}_gp_{}_nunique'.format(i, j)], axis=1, inplace=True)

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
    train['tlsVersion_map'] = train['tlsVersion'].map(tlsVersion_map)
    cat_cols.append('tlsVersion_map')

    cat_cols += ['srcAddressPort', 'destAddressPort']
    num_cols = ['bytesOut', 'bytesIn', 'pktsIn', 'pktsOut']

    for i in num_cols:
        train[i] = np.log1p(train[i])

    count_encode = {}
    for col in cat_cols:
        print(col)
        vc = train[col].value_counts(dropna=True, normalize=True)
        train[col + '_count'] = train[col].map(vc).astype('float32')
        count_encode[col + '_count'] = vc

    # joblib.dump(count_encode, './count_encode.pkl')

    def max_min(s):
        return s.max() - s.min()

    def quantile(s, q=0.25):
        return s.quantile(q)

    cross_encode = {}
    for f1 in cat_cols:
        for f2 in num_cols:
            tmp = train.groupby(f1, as_index=False)[f2].agg({
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
            train = train.merge(tmp, on=f1, how='left')
            cross_encode['{}_stats_{}'.format(f1, f2)] = tmp
    # joblib.dump(cross_encode, './cross_encode.pkl')

    train = arithmetic(train, num_cols)
    used_cols = ['srcPort', 'srcAddressPort_gp_destAddress_nunique_rate', 'srcAddress_bytesOut_max',
                 'srcAddress_bytesOut_min', 'srcAddress_bytesOut_median', 'srcAddress_bytesOut_std',
                 'srcAddress_bytesIn_min', 'srcAddress_bytesIn_skew', 'srcAddress_bytesIn_std',
                 'srcAddress_pktsIn_skew', 'srcAddress_pktsOut_mean', 'srcAddress_pktsOut_skew',
                 'destAddress_bytesOut_min', 'destAddress_bytesIn_max', 'destAddress_bytesIn_median',
                 'destAddress_pktsIn_sum', 'destAddress_pktsOut_quantile_25', 'tlsVersion_bytesOut_min',
                 'tlsVersion_bytesOut_skew', 'tlsSubject_bytesOut_sum', 'tlsSubject_bytesOut_std',
                 'tlsSubject_bytesOut_nunique', 'tlsSubject_bytesOut_quantile_25', 'tlsSubject_bytesIn_max',
                 'tlsSubject_bytesIn_min', 'tlsSubject_bytesIn_std', 'tlsSubject_bytesIn_max_min',
                 'tlsSubject_bytesIn_quantile_25', 'tlsSubject_pktsIn_max', 'tlsSubject_pktsIn_mean',
                 'tlsSubject_pktsIn_quantile_25', 'tlsSubject_pktsOut_skew', 'tlsSubject_pktsOut_max_min',
                 'tlsSubject_pktsOut_quantile_25', 'tlsIssuerDn_bytesOut_min', 'tlsIssuerDn_bytesOut_std',
                 'tlsIssuerDn_bytesOut_quantile_25', 'tlsIssuerDn_bytesOut_quantile_75', 'tlsIssuerDn_bytesIn_median',
                 'tlsIssuerDn_bytesIn_skew', 'tlsIssuerDn_bytesIn_std', 'tlsIssuerDn_bytesIn_quantile_25',
                 'tlsIssuerDn_pktsIn_min', 'tlsIssuerDn_pktsIn_mean', 'tlsIssuerDn_pktsIn_sum',
                 'tlsIssuerDn_pktsIn_skew', 'tlsIssuerDn_pktsIn_nunique', 'tlsIssuerDn_pktsIn_quantile_25',
                 'tlsIssuerDn_pktsIn_quantile_75', 'tlsIssuerDn_pktsOut_skew', 'tlsIssuerDn_pktsOut_std',
                 'tlsIssuerDn_pktsOut_nunique', 'tlsIssuerDn_pktsOut_max_min', 'tlsIssuerDn_pktsOut_quantile_75',
                 'tlsSni_bytesOut_min', 'tlsSni_bytesOut_median', 'tlsSni_bytesOut_nunique', 'tlsSni_bytesOut_max_min',
                 'tlsSni_bytesOut_quantile_25', 'tlsSni_bytesIn_max', 'tlsSni_bytesIn_median', 'tlsSni_bytesIn_mean',
                 'tlsSni_bytesIn_sum', 'tlsSni_bytesIn_std', 'tlsSni_bytesIn_quantile_25', 'tlsSni_bytesIn_quantile_75',
                 'tlsSni_pktsIn_min', 'tlsSni_pktsIn_median', 'tlsSni_pktsIn_mean', 'tlsSni_pktsIn_skew',
                 'tlsSni_pktsIn_quantile_25', 'tlsSni_pktsOut_max', 'tlsSni_pktsOut_nunique', 'tlsSni_pktsOut_max_min',
                 'destAddressPort_pktsOut_mean', 'bytesOut_pktsInc_multiply', 'bytesOut_pktsOut_subtract',
                 'bytesIn_pktsIn_subtract', 'bytesIn_pktsOut_add', 'bytesIn_pktsOut_subtract',
                 'pktsIn_pktsOut_subtract', 'bytesIn_bytesOut_ratio', 'pktsIn_pktsOut_ratio', 'pktsOut_pktsIn_ratio']

    train = train[used_cols]
    train_model = lgb.Booster(model_file='../train_code/lgb.txt')
    train_model.save_model('./lgb.txt')

    pred = train_model.predict(train)
    y_pred = np.where(pred >= 0.11, 1, 0)

    submission['label'] = y_pred
    print('y_pred.mean(): \n', y_pred.mean())

    submission.to_csv(save_path + '机器不学习原子弹也不学习_eta_submission_1030.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    test_path = '../data/test_1.csv'
    sava_path = '../result/'
    test_func(test_path, sava_path)
