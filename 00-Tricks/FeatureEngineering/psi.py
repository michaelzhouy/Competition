# -*- coding: utf-8 -*-
# @Time     : 2020/5/13 19:41
# @Author   : Michael_Zhouy

import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from joblib import Parallel, delayed


# 调用方法
# psi_res = Parallel(n_jobs=4)(delayed(get_psi)(c) for c in tqdm(features))
# psi_df = pd.concat(psi_res)
# features = list(psi_df[psi_df['PSI'] <= 0.2]['变量名'].values)
def get_psi(c):
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


def data_month_psi(df, nameks):
    # nameks是特征名字
    # 为了计算psi
    labels = ['c' + str(i) for i in range(10)]
    # True_out,bins=pd.qcut(df['result'],q=10,retbins=True,labels=labels, duplicates='drop')
    True_out, bins = pd.cut(df['result'], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                            retbins=True,
                            labels=labels)
    df['True_out'] = True_out
    # bins[0] = bins[0]-0.001 # cut左开右闭，之前最小值再分组后组记号为空，这里减0.01划到最左侧区间

    re_total = pd.DataFrame(columns=['月份', 'features', '区间', '基准数', '当月数', '基准占比', '当月占比', 'sub', 'ln', 'PSI'])
    for i in range(0, len(df.月份.unique())):
        # 以前两个月为基准
        fri_m = df.月份.unique()[0]
        sce_m = df.月份.unique()[1]
        m = df.月份.unique()[i]
        data_ks_last = df.loc[(df['月份'] == fri_m) | (df['月份'] == sce_m), ]
        data_ks = df.loc[df['月份'] == m, ]

        a = pd.DataFrame(data_ks_last.True_out.value_counts()).rename(columns={'True_out': '基准占比'})
        a = a.applymap(lambda y: y / sum(a.基准占比))

        b = pd.DataFrame(data_ks.True_out.value_counts()).rename(columns={'True_out': '当月占比'})
        b = b.applymap(lambda y: y / sum(b.当月占比))

        re = pd.merge(a, b, left_index=True, right_index=True)
        re['月份'] = m
        re['基准数'] = data_ks_last.True_out.value_counts()
        re['当月数'] = data_ks.True_out.value_counts()

        psi = 0
        ln = []
        for j in range(len(re)):
            if re['基准占比'][j] == 0:
                re['基准占比'][j] = 0.000001
            if re['当月占比'][j] == 0:
                re['当月占比'][j] = 0.000001
            l = math.log((re['当月占比'][j] / re['基准占比'][j]))
            p = ((re['当月占比'][j] - re['基准占比'][j]) * (math.log((re['当月占比'][j] / re['基准占比'][j]))))

            ln.append(l)
            psi = psi + p

        re['sub'] = re['当月占比'] - re['基准占比']
        re['ln'] = ln
        re['PSI'] = psi
        re['区间'] = re.index
        re['features'] = nameks
        re = re[['月份', 'features', '区间', '基准数', '当月数', '基准占比', '当月占比', 'sub', 'ln', 'PSI']].sort_index(by=["区间"],
                                                                                                       ascending=[True])
        re_total = pd.concat([re_total, re])

    return re_total
