# -*- coding: utf-8 -*-
# @Time     : 2020/11/1 18:16
# @Author   : Michael_Zhouy
import numpy as np
import pandas as pd
import lightgbm as lgb


def get_psi(c, x_train, x_test):
    """
    # 调用方法
    psi_res = Parallel(n_jobs=4)(delayed(get_psi)(c, train, test) for c in used_cols)
    psi_df = pd.concat(psi_res)
    psi_used_cols = list(psi_df[psi_df['PSI'] <= 0.2]['变量名'].values)
    psi_not_used_cols = list(psi_df[psi_df['PSI'] > 0.2]['变量名'].values)
    print('PSI used features: \n', psi_used_cols)
    print('PSI drop features: \n', psi_not_used_cols)
    print('Error drop features: \n', list(set(used_cols) - set(psi_used_cols)))
    @param c:
    @param x_train:
    @param x_test:
    @return:
    """
    psi_res = pd.DataFrame()
    psi_dict={}
    # for c in tqdm(f_cols):
    try:
        t_train = x_train[c].fillna(-998)
        t_test = x_test[c].fillna(-998)
        #获取切分点
        bins=[]
        for i in np.arange(0,1.1,0.2):
            bins.append(t_train.quantile(i))
        bins=sorted(set(bins))
        bins[0]=-np.inf
        bins[-1]=np.inf
        #计算psi
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
        try:
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
        except:
            print('Error: ', i)
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
    return list(col_corr)