import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve, f1_score
import gc


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


# 自定义评估函数（特定阈值下的f1）
def self_metric(preds, train_data):
    labels = train_data.get_label()
    y_preds = np.where(preds >= np.percentile(preds, 95), 1, 0)
    f1 = f1_score(labels, y_preds)
    return 'self_metric', f1, True


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


# 单特征AUC筛选
useful_cols = []
useless_cols = []

for i in train_cols:
    print(i)

    lgb_train = lgb.Dataset(X_train[[i]].values, y_train)
    lgb_valid = lgb.Dataset(X_valid[[i]].values, y_valid, reference=lgb_train)
    lgb_test = lgb.train(params,
                         lgb_train,
                         num_boost_round=1000,
                         valid_sets=[lgb_valid, lgb_train],
                         early_stopping_rounds=50,
                         verbose_eval=20)

    print('*' * 5)
    print(lgb_test.best_score['valid_0']['auc'])
    if lgb_test.best_score['valid_0']['auc'] > 0.52:
        useful_cols.append(i)
    else:
        useless_cols.append(i)
    print('*' * 20)
    print('\n')


def correlation(df, threshold):
    """
    去除特征相关系数大于阈值的特征
    :param df:
    :param threshold:
    :return:
    """
    col_corr = set()
    corr_matrix = df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colName_i = corr_matrix.columns[i]
                colName_j = corr_matrix.columns[j]
                if useful_cols[colName_i] >= useful_cols[colName_j]:
                    col_corr.add(colName_j)
                else:
                    col_corr.add(colName_i)

    return col_corr


col = correlation(df_train.drop(['phone_no_m', 'label'], axis=1), 0.98)
print('Correlated columns: ', col)

oof = []
prediction = df_test[['phone_no_m', 'arpu_202004']]
prediction['label'] = 0
df_importance_list = []

kfold = StratifiedKFold(n_splits=5)
for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(df_train[feature_names], df_train[ycol])):
    print('\nFold_{} Training ================================\n'.format(fold_id + 1))

    X_train = df_train.iloc[trn_idx][feature_names]
    Y_train = df_train.iloc[trn_idx][ycol]

    X_val = df_train.iloc[val_idx][feature_names]
    Y_val = df_train.iloc[val_idx][ycol]

    lgb_train = lgb.Dataset(X_train, Y_train)
    lgb_valid = lgb.Dataset(X_val, Y_val, reference=lgb_train)

    lgb_model = lgb.train(params,
                          lgb_train,
                          num_boost_round=10000,
                          valid_sets=[lgb_valid, lgb_train],
                          early_stopping_rounds=100,
                          verbose_eval=10)

    pred_val = lgb_model.predict(X_val)
    df_oof = df_train.iloc[val_idx][['phone_no_m', ycol]].copy()
    df_oof['pred'] = pred_val
    oof.append(df_oof)

    pred_test = lgb_model.predict(df_test[feature_names])
    prediction['label_{}'.format(fold_id)] = pred_test

    importance = lgb_model.feature_importance(importance_type='gain')
    feature_name = lgb_model.feature_name()
    df_importance = pd.DataFrame({
        'feature_name': feature_name,
        'importance': importance
    })
    df_importance_list.append(df_importance)

    del lgb_model, pred_val, pred_test, X_train, Y_train, X_val, Y_val
    gc.collect()


df_importance = pd.concat(df_importance_list)
df_importance = df_importance.groupby(['feature_name'])['importance'].agg(
    'mean').sort_values(ascending=False).reset_index()