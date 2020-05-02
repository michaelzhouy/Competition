# -*- coding: utf-8 -*-
# @Time     : 2020/5/2 11:06
# @Author   : Michael_Zhouy

import xgboost as xgb

means = X_tr.groupby(col).target.mean()
train_new[col + '_mean_target'] = train_new[col].map(means)
val_new[col + '_mean_target'] = val_new[col].map(means)

dtrain = xgb.DMatrix(train_new, label=y_tr)
dvalid = xgb.DMatrix(val_new, label=y_val)

evallist = [(dtrain, 'train'), (dvalid, 'eval')]
evals_result = {}
model = xgb.train(xgb_par,
                  dtrain,
                  3000,
                  evals=evallist,
                  verbose_eval=30,
                  evals_result=evals_result,
                  early_stopping_rounds=50)
