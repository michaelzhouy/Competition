!rm -r /home/workspace/data
!ls /home/workspace/input/*/*.zip | xargs -n1 unzip -d /home/workspace/data

!pip install lightgbm

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import lightgbm as lgb

payment_a = pd.read_csv('/home/workspace/data/fk_train/payment_a.csv')
orders_a = pd.read_csv('/home/workspace/data/fk_train/orders_a.csv')
iot_a = pd.read_csv('/home/workspace/data/fk_train/iot_a.csv')

payment_a = payment_a.loc[payment_a['Y'].notnull(), :]

payment_a['DLSBH'] = payment_a['DLSBH'].map(lambda x: int(x[-2:]))
payment_a['Y'] = payment_a['Y'].map(int)

payment_a_train = payment_a.loc[payment_a['SSMONTH'] != 201904, :]
payment_a_valid = payment_a.loc[payment_a['SSMONTH'] == 201904, :]

y_train = payment_a_train['Y']
X_train = payment_a_train.drop(['device_code', 'customer_id', 'Y'], axis=1)
y_valid = payment_a_valid['Y']
X_valid = payment_a_valid.drop(['device_code', 'customer_id', 'Y'], axis=1)

cat_cols = ['DLSBH', 'SSMONTH']

dtrain = lgb.Dataset(X_train, y_train)
dvalid = lgb.Dataset(X_valid, y_valid, reference=dtrain)

params = {
    'objective': 'binary',
    'boosting': 'gbdt',
#     'metric': 'auc',
    'metric': 'None',  # 用自定义评估函数是将metric设置为'None'
    'learning_rate': 0.1,
    'num_leaves': 31,
    'lambda_l1': 0,
    'lambda_l2': 1,
    'num_threads': 23,
    'min_data_in_leaf': 20,
    'first_metric_only': True,
    'is_unbalance': True,
    'max_depth': -1,
    'seed': 2020
}

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.where(y_hat > 0.5, 1, 0)
    return 'f1', f1_score(y_true, y_hat), True

model = lgb.train(
    params,
    dtrain,
    valid_sets=[dvalid, dtrain],
    num_boost_round=1000000,
    early_stopping_rounds=200,
    verbose_eval=300,
    feval=lgb_f1_score
)

!rm -r /home/workspace/project/model
!mkdir /home/workspace/project/model

model.save_model('/home/workspace/project/model/lgb.txt')