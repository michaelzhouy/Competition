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

lgb.train(params,
          lgb_train,
          valid_sets=lgb_test,
          early_stopping_rounds=200,
          verbose_eval=300)
