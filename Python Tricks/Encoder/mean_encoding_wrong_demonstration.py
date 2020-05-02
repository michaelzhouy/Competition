# -*- coding: utf-8 -*-
# @Time     : 2020/5/2 11:06
# @Author   : Michael_Zhouy

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

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


# CV loop
y_tr = df_tr['traget'].values  # target variable
skf = StratifiedKFold(y_tr, 5, shuffle=True, random_state=123)

for tr_ind, val_ind in skf:
    X_tr, X_val = df_tr.iloc[tr_ind], df_tr.iloc[val_ind]
    for col in cols:
        means = X_val[col].map(X_tr.groupby(col).target.mean())
        X_val[col + '_mean_target'] = means
    train_new.iloc[val_ind] = X_val

prior = df_tr['target'].mean()  # global mean
train_new.fillna(prior, inplace=True)


# Expanding mean
cumsum = df_tr.groupby(col)['target'].cumsum() - df_tr['target']
cumcnt = df_tr.groupby(col).cumcount()
train_new[col + '_mean_target'] = cumsum / cumcnt