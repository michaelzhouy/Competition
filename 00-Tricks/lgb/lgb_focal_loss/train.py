# -*- coding: utf-8 -*-
# @Time     : 2020/7/9 16:51
# @Author   : Michael_Zhouy

import lightgbm as lgb
from .metrics import focal_loss_lgb, focal_loss_lgb_eval_error


lgtrain = lgb.Dataset(X_tr, y_tr,
                      feature_name=colnames,
                      categorical_feature=categorical_columns,
                      free_raw_data=False)
lgvalid = lgtrain.create_valid(X_val, y_val)

focal_loss = lambda x,y: focal_loss_lgb(x, y, 0.25, 2.)
eval_error = lambda x,y: focal_loss_lgb_eval_error(x, y, 0.25, 2.)
params = {
    'learning_rate': 0.1,
    'num_boost_round': 10
}
model = lgb.train(params,
                  lgtrain,
                  valid_sets=[lgvalid],
                  fobj=focal_loss,
                  feval=eval_error)
