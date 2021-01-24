# -*- coding:utf-8 -*-
# Time   : 2021/1/24 22:26
# Email  : 15602409303@163.com
# Author : Zhou Yang
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import gc

train = data[data['Label'] != -1]
target = train.pop('Label')
test = data[data['Label'] == -1]
test.drop(['Label'], axis=1, inplace=True)

# 划分数据集
print('划分数据集...')
x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=2018)

print('开始训练gbdt..')
gbm = lgb.LGBMRegressor(
    objective='binary',
    subsample=0.8,
    min_child_weight=0.5,
    colsample_bytree=0.7,
    num_leaves=100,
    max_depth=12,
    learning_rate=0.05,
    n_estimators=10
)

gbm.fit(x_train, y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        eval_names=['train', 'val'],
        eval_metric='binary_logloss')

model = gbm.booster_
gbdt_feats_train = model.predict(train, pred_leaf=True)
gbdt_feats_test = model.predict(test, pred_leaf=True)
gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(gbdt_feats_train.shape[1])]
df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns=gbdt_feats_name)
df_test_gbdt_feats = pd.DataFrame(gbdt_feats_test, columns=gbdt_feats_name)

train = pd.concat([train, df_train_gbdt_feats], axis=1)
test = pd.concat([test, df_test_gbdt_feats], axis=1)
train_len = train.shape[0]
data = pd.concat([train, test])
del train
del test
gc.collect()

print('one-hot features for leaf node')
for col in gbdt_feats_name:
    print('feature:', col)
    onehot_feats = pd.get_dummies(data[col], prefix=col)
    data.drop([col], axis=1, inplace=True)
    data = pd.concat([data, onehot_feats], axis=1)
print('one-hot结束')

train = data[: train_len]
test = data[train_len:]
del data
gc.collect()

x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.3, random_state=2018)
lr = LogisticRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict_proba(test)[:, 1]
