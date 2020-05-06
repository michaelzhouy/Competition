import pandas as pd
import lightgbm as lgb

params = {'boosting_type': 'gbdt',
          'objective': 'regression',
          'metric': 'mae',
          'learning_rate': 0.05}

params = {'objective': 'binary',
          'boosting': 'gbdt',
          'metric': 'auc',
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

lgb_train = lgb.Dataset(train_x, label=train_y)
lgb_test = lgb.Dataset(test_x, label=test_y, reference=lgb_train)

# 模型训练
lgb_model = lgb.train(params,
                      lgb_train,
                      valid_sets=lgb_test,
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
