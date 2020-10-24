# -*- coding: utf-8 -*-
# @Time     : 2020/5/13 19:41
# @Author   : Michael_Zhouy

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


def get_psi(c, x_train, x_test):
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


# 调用方法
psi_res = Parallel(n_jobs=4)(delayed(get_psi)(c, X, X_test) for c in used_cols)
psi_df = pd.concat(psi_res)
psi_used_cols = list(psi_df[psi_df['PSI'] <= 0.2]['变量名'].values)
psi_not_used_cols = list(psi_df[psi_df['PSI'] > 0.2]['变量名'].values)
print('PSI used features: \n', psi_used_cols)
print('PSI drop features: \n', psi_not_used_cols)
print('Error drop features: \n', list(set(used_cols) - set(psi_used_cols)))
