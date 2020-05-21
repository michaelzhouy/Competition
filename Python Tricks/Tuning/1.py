# -*- coding: utf-8 -*-
# @Time     : 2020/5/21 15:08
# @Author   : Michael_Zhouy

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import KFold
from hyperopt import fmin, tpe, hp  # ,rand#,pyll#,partial

def get_set(n1,data='trained.csv',n_splits=10,y=False,random_state=0):
    data=pd.read_csv(data)
    kf = KFold(n_splits=n_splits,shuffle=True,random_state=random_state)
    if y:
        train,test=pd.DataFrame(),pd.DataFrame()
        clas=list(data[y].unique())
        for cla in clas:
            i=0
            dd=data[data[y]==cla]
            for train_index,test_index in kf.split(dd):
            i=i+1
            if n1==i:
            train=train.append(data.loc[list(train_index)])
            test=test.append(data.loc[list(test_index)])
    else:
        i=0
        for train_index,test_index in kf.split(data):
            i=i+1
            if n1==i:
                train=data.iloc[list(train_index),:]
                test=data.iloc[list(test_index),:]
    return train,test


def scorer(yp,data):
    yt= data.get_label()
    score=auc(yt,yp)
    return 'auc',score,True


def peropt(param):
    conf=['num_leaves','max_depth','min_child_samples','max_bin']
    for i in conf:
        param[i]=int(param[i])
    evals_result={}
    lgb.train(param,
              dtrain,
              2000,
              feval=scorer,
              valid_sets=[dval],
              verbose_eval=None,
              evals_result=evals_result,
              early_stopping_rounds=10)
    best_score=evals_result['valid_0']['auc'][-11]
    #print(param,best_score,len(evals_result['valid_0']['auc'])-10)
    result.append((param, best_score, len(evals_result['valid_0']['auc'])-10))
    return -best_score


if 0:#数据集
    i=1
    x_train,x_test=get_set(i,n_splits=5)
    x_train.pop('CaseId')
    x_test.pop('CaseId')
    y_train=x_train.pop('Evaluation')
    y_test=x_test.pop('Evaluation')
    dtrain=lgb.Dataset(x_train,y_train)
    dval=lgb.Dataset(x_test,y_test)

if 1:#调参
    space={'num_leaves': hp.quniform('num_leaves',50,70,1),
           'max_depth': hp.quniform('max_depth',7,15,1),
           'min_child_samples': hp.quniform('min_child_samples',5,20,1),
           'max_bin': hp.quniform('max_bin',100,150,5),
           'learning_rate': hp.choice('learning_rate',[0.01]),
           'subsample': hp.uniform('subsample',0.9,1),
           'colsample_bytree': hp.uniform('colsample_bytree',0.95,1),
           'min_split_gain': hp.loguniform('min_split_gain',-5,2),
           'reg_alpha': hp.loguniform('reg_alpha',-5,2),
           'reg_lambda': hp.loguniform('reg_lambda',-5,2)}
    result=[]
    #print(pyll.stochastic.sample(space))#抽样
    algo=partial(tpe.suggest,n_startup_jobs=10)#作用未知
    fmin(peropt, space=space, algo=tpe.suggest, max_evals=100)
    sort=sorted(result, key=lambda x:x[1], reverse=True)