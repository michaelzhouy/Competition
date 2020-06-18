# -*- coding: utf-8 -*-
# @Time     : 2020/5/21 11:30
# @Author   : Michael_Zhouy

from sklearn.model_selection import cross_val_score
import lightgbm as lgb
from hyperopt import fmin, tpe, hp, partial

space = {'max_depth': hp.randint('max_depth', 15),
         'n_estimators': hp.randint('n_estimators', 10),  # [0,1,2,3,4,5] -> [50,]
         'learning_rate': hp.randint('learning_rate', 6),  # [0,1,2,3,4,5] -> 0.05,0.06
         'subsample': hp.randint('subsample', 4),  # [0,1,2,3] -> [0.7,0.8,0.9,1.0]
         'min_child_weight': hp.randint('min_child_weight', 5)}


# 分类
def GBM(argsDict):
    max_depth = argsDict['max_depth'] + 5
    learning_rate = argsDict['learning_rate'] * 0.02 + 0.05
    subsample = argsDict['subsample'] * 0.1 + 0.7
    print('max_depth:' + str(max_depth))
    print('learning_rate:' + str(learning_rate))
    print('subsample:' + str(subsample))

    gbm = lgb.LGBMClassifier(max_depth=max_depth,
                             learning_rate=learning_rate,
                             subsample=subsample,
                             num_leaves=31,
                             objective='binary',
                             is_unbalance=True,
                             seed=2020)

    metric = cross_val_score(gbm, X_train, y_train, cv=5, scoring='roc_auc').mean()
    print(metric)
    return -metric


algo = partial(tpe.suggest, n_startup_jobs=1)
best = fmin(GBM, space, algo=algo, max_evals=4)  # max_evals表示想要训练的最大模型数量，越大越容易找到最优解

print('-' * 20)
print(best)
print(GBM(best))
