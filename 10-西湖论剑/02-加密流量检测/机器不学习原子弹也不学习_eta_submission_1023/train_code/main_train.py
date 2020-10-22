# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import lightgbm as lgb


def count_encode(df, cols=[], path='./'):
    """
    count编码
    @param df:
    @param cols:
    @param path:
    @return:
    """
    for col in cols:
        print(col)
        vc = df[col].value_counts(dropna=True, normalize=True)
        df[col + '_count'] = df[col].map(vc.to_dict()).astype('float32')
        print(vc.to_dict())
        np.save(path + '{}_count.npy'.format(col), vc.to_dict())
    return df


def cross_cat_num(df, cat_col, num_col, path='./'):
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
            tmp.to_csv(path + '{}_{}.csv'.format(f1, f2), index=False)
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


def get_psi(c, x_train, x_test):
    psi_res = pd.DataFrame()
    psi_dict={}
    # for c in tqdm(f_cols):
    try:
        t_train = x_train[c].fillna(-998)
        t_test = x_test[c].fillna(-998)
        # 获取切分点
        bins=[]
        for i in np.arange(0,1.1,0.2):
            bins.append(t_train.quantile(i))
        bins=sorted(set(bins))
        bins[0]=-np.inf
        bins[-1]=np.inf
        # 计算psi
        t_psi = pd.DataFrame()
        t_psi['train'] = pd.cut(t_train,bins).value_counts().sort_index()
        t_psi['test'] = pd.cut(t_test,bins).value_counts()
        t_psi.index=[str(x) for x in t_psi.index]
        t_psi.loc['总计',:] = t_psi.sum()
        t_psi['train_rate'] = t_psi['train']/t_psi.loc['总计','train']
        t_psi['test_rate'] = t_psi['test']/t_psi.loc['总计','test']
        t_psi['psi'] = (t_psi['test_rate']-t_psi['train_rate'])*(np.log(t_psi['test_rate'])-np.log(t_psi['train_rate']))
        t_psi.loc['总计','psi'] = t_psi['psi'].sum()
        t_psi.index.name=c
        #汇总
        t_res = pd.DataFrame([[c,t_psi.loc['总计','psi']]],
                             columns=['变量名','PSI'])
        psi_res = pd.concat([psi_res,t_res])
        psi_dict[c]=t_psi
        print(c,'done')
    except:
        print(c,'error')
    return psi_res #, psi_dict


def auc_select(X_train, y_train, X_valid, y_valid, cols, threshold=0.52):
    """
    基于AUC的单特征筛选
    @param X_train:
    @param y_train:
    @param X_valid:
    @param y_valid:
    @param cols:
    @param threshold:
    @return:
    """
    useful_dict = dict()
    useless_dict = dict()
    params = {
        'objective': 'binary',
        'boosting': 'gbdt',
        'metric': 'auc',
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
        lgb_model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_valid, lgb_train],
            num_boost_round=1000,
            early_stopping_rounds=50,
            verbose_eval=500
        )
        print('*' * 10)
        print(lgb_model.best_score['valid_0']['auc'])
        if lgb_model.best_score['valid_0']['auc'] > threshold:
            useful_dict[i] = lgb_model.best_score['valid_0']['auc']
        else:
            useless_dict[i] = lgb_model.best_score['valid_0']['auc']
    useful_cols = list(useful_dict.keys())
    useless_cols = list(useless_dict.keys())
    return useful_dict, useless_dict, useful_cols, useless_cols


def correlation(df, useful_dict, threshold=0.98):
    """
    去除特征相关系数大于阈值的特征
    @param df:
    @param threshold:
    @param useful_dict:
    @return:
    """
    col_corr = set()
    corr_matrix = df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colName_i = corr_matrix.columns[i]
                colName_j = corr_matrix.columns[j]
                if useful_dict[colName_i] >= useful_dict[colName_j]:
                    col_corr.add(colName_j)
                else:
                    col_corr.add(colName_i)
    return col_corr


def train_func(train_path):
    # 请填写训练代码
    train = pd.read_csv(train_path)
    drop_cols = ['appProtocol', 'srcAddress', 'destAddress', 'tlsSubject', 'tlsIssuerDn', 'tlsSni', 'srcPort', 'destPort']
    train.drop(drop_cols, axis=1, inplace=True)

    # train['srcAddressPort'] = train['srcAddress'].astype(str) + train['srcPort'].astype(str)
    # train['destAddressPort'] = train['destAddress'].astype(str) + train['destPort'].astype(str)

    # # srcAddress To destAddress
    # tmp = train.groupby('srcAddress', as_index=False)['destAddress'].agg({
    #     's2d_count': 'count',
    #     's2d_nunique': 'nunique'
    # })
    # train = train.merge(tmp, on='srcAddress', how='left')
    #
    # # srcAddressPort To destAddressPort
    # tmp = train.groupby('srcAddressPort', as_index=False)['destAddressPort'].agg({
    #     'sp2dp_count': 'count',
    #     'sp2dp_nunique': 'nunique'
    # })
    # train = train.merge(tmp, on='srcAddressPort', how='left')
    #
    # # srcAddress To destAddressPort
    # tmp = train.groupby('srcAddress', as_index=False)['destAddressPort'].agg({
    #     's2dp_count': 'count',
    #     's2dp_nunique': 'nunique'
    # })
    # train = train.merge(tmp, on='srcAddress', how='left')
    #
    # # srcAddressPort To destAddress
    # tmp = train.groupby('srcAddressPort', as_index=False)['destAddress'].agg({
    #     'sp2d_count': 'count',
    #     'sp2d_nunique': 'nunique'
    # })
    # train = train.merge(tmp, on='srcAddressPort', how='left')
    #
    # # destAddress To srcAddress
    # tmp = train.groupby('destAddress', as_index=False)['srcAddress'].agg({
    #     'd2s_count': 'count',
    #     'd2s_nunique': 'nunique'
    # })
    # train = train.merge(tmp, on='destAddress', how='left')
    #
    # # destAddressPort To srcAddressPort
    # tmp = train.groupby('destAddressPort', as_index=False)['srcAddressPort'].agg({
    #     'dp2sp_count': 'count',
    #     'dp2sp_nunique': 'nunique'
    # })
    # train = train.merge(tmp, on='destAddressPort', how='left')
    #
    # # destAddressPort To srcAddress
    # tmp = train.groupby('destAddressPort', as_index=False)['srcAddress'].agg({
    #     'dp2s_count': 'count',
    #     'dp2s_nunique': 'nunique'
    # })
    # train = train.merge(tmp, on='destAddressPort', how='left')
    #
    # # destAddress To srcAddressProt
    # tmp = train.groupby('destAddress', as_index=False)['srcAddressPort'].agg({
    #     'd2sp_count': 'count',
    #     'd2sp_nunique': 'nunique'
    # })
    # train = train.merge(tmp, on='destAddress', how='left')

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
    train['tlsVersion1'] = train['tlsVersion'].map(tlsVersion_map1)

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
    train['tlsVersion2'] = train['tlsVersion'].map(tlsVersion_map2)
    cat_cols = ['tlsVersion1', 'tlsVersion2']

    # cat_cols += ['srcAddressPort', 'destAddressPort']
    num_cols = ['bytesOut', 'bytesIn', 'pktsIn', 'pktsOut']

    train = arithmetic(train, num_cols)
    train = count_encode(train, cat_cols)
    train = cross_cat_num(train, cat_cols, num_cols)

    used_cols = [i for i in train.columns if i not in ['eventId', 'label', 'tlsVersion']]
    y = train['label']
    train = train[used_cols].copy()

    X_train, X_valid, y_train, y_valid = train_test_split(train, y, test_size=0.25, random_state=2020, stratify=y)

    print('y_train mean: ', y_train.mean())
    print('y_valid mean: ', y_valid.mean())

    used_cols = X_train.columns.to_list()

    useful_dict, useless_dict, useful_cols, useless_cols = auc_select(X_train, y_train, X_valid, y_valid, used_cols, threshold=0.52)
    print('AUC drop features: \n', useless_cols)

    X_train = X_train[useful_cols].copy()
    X_valid = X_valid[useful_cols].copy()

    col_corr = correlation(X_train, useful_dict, threshold=0.98)
    print('Correlation drop features: \n', col_corr)

    X_train.drop(col_corr, axis=1, inplace=True)
    X_valid.drop(col_corr, axis=1, inplace=True)

    used_cols = X_train.columns.to_list()
    print('*' * 20)
    print('used_cols: \n', used_cols)
    print('len(used_cols): \n', len(used_cols))

    train_dataset = lgb.Dataset(X_train, y_train)
    valid_dataset = lgb.Dataset(X_valid, y_valid, reference=train_dataset)
    all_dataset = lgb.Dataset(train[used_cols], y, reference=train_dataset)

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
    valid_model = lgb.train(
        params,
        train_dataset,
        valid_sets=[valid_dataset, train_dataset],
        early_stopping_rounds=200,
        num_boost_round=1000000,
        verbose_eval=300
    )
    pred = valid_model.predict(X_valid)

    f1_best = 0
    for i in np.arange(0.1, 1, 0.01):
        y_valid_pred = np.where(pred > i, 1, 0)
        f1 = np.round(f1_score(y_valid, y_valid_pred), 5)
        if f1 > f1_best:
            threshold = i
            f1_best = f1

    print('threshold: ', threshold)
    np.save('threshold.npy', threshold)
    y_valid_pred = np.where(pred > threshold, 1, 0)
    print('Valid F1: ', np.round(f1_score(y_valid, y_valid_pred), 5))
    print('Valid mean label: ', np.mean(y_valid_pred))

    train_model = lgb.train(
        params,
        all_dataset,
        num_boost_round=valid_model.best_iteration + 20
    )
    train_model.save_model('./lgb.txt')


if __name__ == '__main__':
    train_path = '../data/train.csv'
    train_func(train_path)
