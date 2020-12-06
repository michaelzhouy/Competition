# -*- coding: utf-8 -*-
# @Time     : 2020/8/3 13:42
# @Author   : Michael_Zhouy

import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance


trn_x, trn_y, val_x, val_y = x_train[:nums], y_train[:nums], x_train[nums:], y_train[nums:]

train_matrix = xgb.DMatrix(trn_x, label=trn_y, missing=np.nan)
valid_matrix = xgb.DMatrix(val_x, label=val_y, missing=np.nan)
train_all_matrix = xgb.DMatrix(x_train, y_train, missing=np.nan)
test_matrix = xgb.DMatrix(x_test, label=val_y, missing=np.nan)

params = {
    'booster': 'gbtree',
    'eval_metric': 'mae',
    'min_child_weight': 5,
    'max_depth': 8,
    'subsample': 0.5,
    'colsample_bytree': 0.5,
    'eta': 0.01,
    'seed': 2020,
    'nthread': 36,
    'silent': 1
}

watchlist = [(train_matrix, 'train'), (valid_matrix, 'eval')]

model_eval = xgb.train(params,
                       train_matrix,
                       num_boost_round=50000,
                       evals=watchlist,
                       verbose_eval=500,
                       early_stopping_rounds=1000)
val_pred = model_eval.predict(valid_matrix, ntree_limit=model_eval.best_ntree_limit).reshape(-1, 1)


# 可视化特征重要性
plot_importance(model_eval)
plt.show()

# get_score()返回特征重要性，是一个字典
feature_importance_dict = model_eval.get_score(importance_type='gain')
sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)