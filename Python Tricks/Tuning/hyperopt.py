# -*- coding: utf-8 -*-
# @Time     : 2020/5/20 10:42
# @Author   : Michael_Zhouy

import numpy as np
import lightgbm as lgb
import hyperopt
from hyperopt import fmin, tpe, hp, partial
from sklearn.metrics import mean_squared_error
from numpy.random import RandomState
import warnings
warnings.filterwarnings('ignore')

# 自定义hyperopt的参数空间
space = {'max_depth': hp.randint('max_depth', 15),
         # 'num_trees': hp.randint('num_trees', 300),
         'learning_rate': hp.uniform('learning_rate', 1e-3, 5e-1),
         'bagging_fraction': hp.randint('bagging_fraction', 5),
         'num_leaves': hp.randint('num_leaves', 6)}


def argsDict_tranform(argsDict, isPrint=False):
    argsDict['max_depth'] = argsDict['max_depth'] + 5
    # argsDict['num_trees'] = argsDict['num_trees'] + 150
    argsDict['learning_rate'] = argsDict['learning_rate'] * 0.02 + 0.05
    argsDict['bagging_fraction'] = argsDict['bagging_fraction'] * 0.1 + 0.5
    argsDict['num_leaves'] = argsDict['num_leaves'] * 3 + 10
    if isPrint:
        print(argsDict)
    else:
        pass

    return argsDict


def lightgbm_factory(argsDict):
    argsDict = argsDict_tranform(argsDict)

    params = {'nthread': -1,
              'max_depth': argsDict['max_depth'],
              'num_trees': argsDict['num_trees'],
              'eta': argsDict['learning_rate'],
              'bagging_fraction': argsDict['bagging_fraction'],
              'num_leaves': argsDict['num_leaves'],
              'objective': 'regression',
              'feature_fraction': 0.7,
              'lambda_l1': 0,
              'lambda_l2': 0,
              'bagging_seed': 100,
              'metric': ['rmse']}

    model_lgb = lgb.train(params,
                          train_data,
                          num_boost_round=300,
                          valid_sets=[test_data],
                          early_stopping_rounds=100)

    return get_tranformer_score(model_lgb)


def get_tranformer_score(tranformer):
    model = tranformer
    prediction = model.predict(x_predict,
                               num_iteration=model.best_iteration)

    return mean_squared_error(y_predict, prediction)


# 开始使用hyperopt进行自动调参
algo = partial(tpe.suggest, n_startup_jobs=1)
best = fmin(lightgbm_factory, space, algo=algo, max_evals=20, pass_expr_memo_ctrl=None)


RMSE = lightgbm_factory(best)
print('best :', best)
print('best param after transform :')
argsDict_tranform(best, isPrint=True)
print('rmse of the best lightgbm:', np.sqrt(RMSE))


# 交叉验证调参
train_all_data = lgb.Dataset(data=x_train_all,label=y_train_all)


def hyperopt_objective(params):

    model = lgb.LGBMRegressor(
        num_leaves=31,
        max_depth=int(params['max_depth']) + 5,
        learning_rate=params['learning_rate'],
        objective='regression',
        eval_metric='rmse',
        nthread=-1,
    )

    num_round = 10
    res = lgb.cv(model.get_params(),
                 train_all_data,
                 num_round,
                 nfold=5,
                 metrics='rmse',
                 early_stopping_rounds=10)

    return min(res['rmse-mean'])  # as hyperopt minimises


params_space = {'max_depth': hyperopt.hp.randint('max_depth', 6),
                'learning_rate': hyperopt.hp.uniform('learning_rate', 1e-3, 5e-1)}

trials = hyperopt.Trials()

best = hyperopt.fmin(hyperopt_objective,
                     space=params_space,
                     algo=hyperopt.tpe.suggest,
                     max_evals=50,
                     trials=trials,
                     rstate=RandomState(123))

print('\n展示hyperopt获取的最佳结果，但是要注意的是我们对hyperopt最初的取值范围做过一次转换')
print(best)
# 展示hyperopt获取的最佳结果，但是要注意的是我们对hyperopt最初的取值范围做过一次转换
# {'learning_rate': 0.059356676015000595, 'max_depth': 5}