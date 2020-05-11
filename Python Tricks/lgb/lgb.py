import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_curve

# regression
params = {'boosting_type': 'gbdt',
          'objective': 'regression',
          'metric': 'mae',
          'learning_rate': 0.05}

# binary
params = {'objective': 'binary',
          'boosting': 'gbdt',
          'metric': 'auc',
          # 'metric': 'None',  # 用自定义评估函数是将metric设置为'None'
          'num_iterations': 1000000,
          'learning_rate': 0.1,
          'num_leaves': 31,
          'lambda_l1': 0,
          'lambda_l2': 1,
          'num_threads': 23,
          'min_data_in_leaf': 20,
          'first_metric_only': True,
          'is_unbalance': True,
          'max_depth': -1,
          'seed': 2020}


# 自定义评估函数
def self_metric(preds, train_data):
    labels = train_data.get_label()
    fpr, tpr, _ = roc_curve(labels, preds)
    max_tpr = tpr[np.where(fpr < 0.001)][-1]

    return 'self_metric', max_tpr, True


lgb_train = lgb.Dataset(train_x, label=train_y)
lgb_test = lgb.Dataset(test_x, label=test_y, reference=lgb_train)

# 模型训练
lgb_model = lgb.train(params,
                      lgb_train,
                      valid_sets=[lgb_test, lgb_train],
                      early_stopping_rounds=200,
                      verbose_eval=300)

# 保存模型
lgb_model.save_model(path)
model = lgb.Booster(model_file=path)

# 导出特征重要性
importance = lgb_model.feature_importance(importance_type='gain')
feature_name = lgb_model.feature_name()

feature_importance = pd.DataFrame({'feature_name': feature_name, 'importance': importance}).sort_values(by='importance',
                                                                                                        ascending=False)
feature_importance.to_csv('feature_importance.csv', index=False)


# Stacking
skf = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
for i, (train_index, test_index) in enumerate(skf.split(train_x, train_y)):
    print(i)
    X_train, X_test, y_train, y_test = train_x.iloc[train_index], train_x.iloc[test_index], train_y.iloc[train_index], train_y.iloc[test_index]
    lgb_train = lgb.Dataset(X_train,
                            label=y_train)
    lgb_test = lgb.Dataset(X_test,
                           label=y_test,
                           reference=lgb_train)
    exec('gbm{} = lgb.train(params, lgb_train, valid_sets=[lgb_test, lgb_test], early_stopping_rounds=200, verbose_eval=100)'.format(i))
