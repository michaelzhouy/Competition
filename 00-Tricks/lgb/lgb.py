import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve, f1_score
import gc


# Stacking
skf = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
for i, (train_index, test_index) in enumerate(skf.split(train_x, train_y)):
    print(i)
    X_train, X_test, y_train, y_test = train_x.iloc[train_index], train_x.iloc[test_index], train_y.iloc[train_index], train_y.iloc[test_index]
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train)
    exec('gbm{} = lgb.train(params, lgb_train, valid_sets=[lgb_test, lgb_test], early_stopping_rounds=200, verbose_eval=100)'.format(i))
