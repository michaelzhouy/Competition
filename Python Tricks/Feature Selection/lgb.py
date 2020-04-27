import lightgbm as lgb

params = {'boosting_type': 'gbdt',
          'objective': 'regression',
          'metric': 'mae',
          'learning_rate': 0.05}
		  

