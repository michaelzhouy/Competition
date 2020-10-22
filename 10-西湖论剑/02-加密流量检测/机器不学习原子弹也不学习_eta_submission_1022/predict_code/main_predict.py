# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb


def count_encode(df, cols=[], path='../train_code/'):
    """
    count编码
    @param df:
    @param cols:
    @return:
    """
    for col in cols:
        d = np.load(path + '{}_count.npy'.format(col), allow_pickle=True).item()
        df[col + '_count'] = df[col].map(d)
    return df


def cross_cat_num(df, cat_col, num_col, path='../train_code/'):
    """
    类别特征与数据特征groupby统计
    @param df:
    @param cat_col: 类别特征
    @param num_col: 数值特征
    @return:
    """
    for f1 in cat_col:
        for f2 in num_col:
            tmp = pd.read_csv(path + '{}_{}.csv'.format(f1, f2))
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

    drop_cols = ['appProtocol', 'srcAddress', 'destAddress', 'tlsSubject', 'tlsIssuerDn', 'tlsSni', 'srcPort',
                 'destPort']
    test.drop(drop_cols, axis=1, inplace=True)

    # cat_cols = ['srcAddress', 'destAddress',
    #             'tlsVersion', 'tlsSubject', 'tlsIssuerDn', 'tlsSni']
    # 
    # test['srcAddressPort'] = test['srcAddress'].astype(str) + test['srcPort'].astype(str)
    # test['destAddressPort'] = test['destAddress'].astype(str) + test['destPort'].astype(str)
    # 
    # # srcAddress To destAddress
    # tmp = test.groupby('srcAddress', as_index=False)['destAddress'].agg({
    #     's2d_count': 'count',
    #     's2d_nunique': 'nunique'
    # })
    # test = test.merge(tmp, on='srcAddress', how='left')
    # 
    # # srcAddressPort To destAddressPort
    # tmp = test.groupby('srcAddressPort', as_index=False)['destAddressPort'].agg({
    #     'sp2dp_count': 'count',
    #     'sp2dp_nunique': 'nunique'
    # })
    # test = test.merge(tmp, on='srcAddressPort', how='left')
    # 
    # # srcAddress To destAddressPort
    # tmp = test.groupby('srcAddress', as_index=False)['destAddressPort'].agg({
    #     's2dp_count': 'count',
    #     's2dp_nunique': 'nunique'
    # })
    # test = test.merge(tmp, on='srcAddress', how='left')
    # 
    # # srcAddressPort To destAddress
    # tmp = test.groupby('srcAddressPort', as_index=False)['destAddress'].agg({
    #     'sp2d_count': 'count',
    #     'sp2d_nunique': 'nunique'
    # })
    # test = test.merge(tmp, on='srcAddressPort', how='left')
    # 
    # # destAddress To srcAddress
    # tmp = test.groupby('destAddress', as_index=False)['srcAddress'].agg({
    #     'd2s_count': 'count',
    #     'd2s_nunique': 'nunique'
    # })
    # test = test.merge(tmp, on='destAddress', how='left')
    # 
    # # destAddressPort To srcAddressPort
    # tmp = test.groupby('destAddressPort', as_index=False)['srcAddressPort'].agg({
    #     'dp2sp_count': 'count',
    #     'dp2sp_nunique': 'nunique'
    # })
    # test = test.merge(tmp, on='destAddressPort', how='left')
    # 
    # # destAddressPort To srcAddress
    # tmp = test.groupby('destAddressPort', as_index=False)['srcAddress'].agg({
    #     'dp2s_count': 'count',
    #     'dp2s_nunique': 'nunique'
    # })
    # test = test.merge(tmp, on='destAddressPort', how='left')
    # 
    # # destAddress To srcAddressProt
    # tmp = test.groupby('destAddress', as_index=False)['srcAddressPort'].agg({
    #     'd2sp_count': 'count',
    #     'd2sp_nunique': 'nunique'
    # })
    # test = test.merge(tmp, on='destAddress', how='left')
    # 
    # tlsVersion_map = {
    #     'TLSv1': 1,
    #     'TLS 1.2': 1,
    #     'TLS 1.3': 1,
    #     'SSLv2': 2,
    #     'SSLv3': 3,
    #     '0x4854': 4,
    #     '0x4752': 4,
    #     'UNDETERMINED': 5
    # }
    # test['tlsVersion_map'] = test['tlsVersion'].map(tlsVersion_map)
    # cat_cols.append('tlsVersion_map')
    # 
    # cat_cols += ['srcAddressPort', 'destAddressPort']
    # num_cols = ['bytesOut', 'bytesIn', 'pktsIn', 'pktsOut']

    tlsVersion_map1 = {
        'TLSv1': 1,
        'TLS 1.2': 2,
        'TLS 1.3': 3,
        'SSLv2': 4,
        'SSLv3': 5,
        '0x4854': 6,
        '0x4752': 6,
        'UNDETERMINED': 7
    }
    test['tlsVersion1'] = test['tlsVersion'].map(tlsVersion_map1)

    tlsVersion_map2 = {
        'TLSv1': 1,
        'TLS 1.2': 1,
        'TLS 1.3': 1,
        'SSLv2': 2,
        'SSLv3': 3,
        '0x4854': 4,
        '0x4752': 4,
        'UNDETERMINED': 5
    }
    test['tlsVersion2'] = test['tlsVersion'].map(tlsVersion_map2)
    cat_cols = ['tlsVersion1', 'tlsVersion2']

    # cat_cols += ['srcAddressPort', 'destAddressPort']
    num_cols = ['bytesOut', 'bytesIn', 'pktsIn', 'pktsOut']

    test = arithmetic(test, num_cols)
    test = count_encode(test, cat_cols)
    test = cross_cat_num(test, cat_cols, num_cols)

    used_cols = [i for i in test.columns if i not in ['eventId', 'tlsVersion']]
    test = test[used_cols].copy()

    print('test.shape: \n', test.shape)
    used_cols = ['pktsIn', 'pktsOut', 'tlsVersion1', 'bytesOut_bytesIn_add', 'bytesOut_bytesIn_subtract', 'bytesOut_bytesInc_multiply', 'bytesOut_pktsIn_add', 'bytesOut_pktsInc_multiply', 'bytesOut_pktsOutc_multiply', 'bytesIn_pktsOut_subtract', 'pktsIn_pktsOut_add', 'pktsIn_pktsOut_subtract', 'bytesOut_bytesIn_ratio', 'bytesOut_pktsIn_ratio', 'bytesOut_pktsOut_ratio', 'bytesIn_bytesOut_ratio', 'bytesIn_pktsIn_ratio', 'bytesIn_pktsOut_ratio', 'pktsIn_bytesOut_ratio', 'pktsIn_bytesIn_ratio', 'pktsIn_pktsOut_ratio', 'pktsOut_bytesOut_ratio', 'pktsOut_bytesIn_ratio', 'pktsOut_pktsIn_ratio', 'tlsVersion1_bytesOut_quantile_25', 'tlsVersion1_bytesIn_quantile_25', 'tlsVersion1_bytesIn_quantile_75', 'tlsVersion1_pktsIn_min', 'tlsVersion1_pktsIn_mean', 'tlsVersion1_pktsIn_quantile_75', 'tlsVersion1_pktsOut_min', 'tlsVersion1_pktsOut_median', 'tlsVersion1_pktsOut_skew', 'tlsVersion1_pktsOut_std', 'tlsVersion1_pktsOut_nunique', 'tlsVersion1_pktsOut_max_min', 'tlsVersion1_pktsOut_quantile_25', 'tlsVersion1_pktsOut_quantile_75']

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
