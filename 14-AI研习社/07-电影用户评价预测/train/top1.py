# -*- coding: utf-8 -*-
# @Time     : 2020/12/8 20:39
# @Author   : Michael_Zhouy

from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import json
import os
import csv
import pickle
from math import exp
import time
import re
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

cols = ['userId', 'movie_id', 'timestamp', 'rating']
cols_test = ['userId', 'movie_id', 'timestamp']
df_x_train = pd.read_csv('../input/train.csv', delimiter=',', names=cols,
                    converters={'userId': str, 'movie_id': str, 'timestamp': str})
df_x_test = pd.read_csv('../input/test.csv', delimiter=',', names=cols_test,
                    converters={'userId': str, 'movie_id': str, 'timestamp': str})

# 增加user和item的其他信息
userInfo = pd.read_csv('../input/userInfo.csv', delimiter=',', converters={'userId': str, 'useGender': str})
itemInfo = pd.read_csv('../input/itemInfo.csv', delimiter=',', converters={'movie_id': str})
# 性别属性数值化
gender_dict = {'M': 0, 'F': 1}
userInfo['useGender'] = userInfo['useGender'].map(gender_dict)
# 邮编处理
userInfo['useZipcode_1'] = userInfo['useZipcode'].str[0:1]
userInfo['useZipcode_2'] = userInfo['useZipcode'].str[0:2]
userInfo['useZipcode_3'] = userInfo['useZipcode'].str[0:3]
# 电影年份和长度
itemInfo['movie_year'] = itemInfo['movie_title'].str[-5:-1]
itemInfo['movie_name_len'] = int(len(itemInfo['movie_title'].str[:])-6)
# 上映年月日
pattern = re.compile(r'^(.*)-(.*)-(.*)$')
itemInfo['release_date'] = itemInfo['release_date'].fillna('0-0-0')
itemInfo['release_d'] = [pattern.match(x).group(1) for x in itemInfo['release_date']]
itemInfo['release_m'] = [pattern.match(x).group(2) for x in itemInfo['release_date']]
itemInfo['release_y'] = [pattern.match(x).group(3) for x in itemInfo['release_date']]
# 评论时间戳
df_x_train['timestamp_2'] = df_x_train['timestamp'].str[0:2]
df_x_train['timestamp_3'] = df_x_train['timestamp'].str[0:3]
df_x_train['timestamp_4'] = df_x_train['timestamp'].str[0:4]
df_x_train['timestamp_5'] = df_x_train['timestamp'].str[0:5]
df_x_train['timestamp_6'] = df_x_train['timestamp'].str[0:6]
df_x_train['timestamp_7'] = df_x_train['timestamp'].str[0:7]
df_x_train['timestamp_8'] = df_x_train['timestamp'].str[0:8]
df_x_test['timestamp_2'] = df_x_test['timestamp'].str[0:2]
df_x_test['timestamp_3'] = df_x_test['timestamp'].str[0:3]
df_x_test['timestamp_4'] = df_x_test['timestamp'].str[0:4]
df_x_test['timestamp_5'] = df_x_test['timestamp'].str[0:5]
df_x_test['timestamp_6'] = df_x_test['timestamp'].str[0:6]
df_x_test['timestamp_7'] = df_x_test['timestamp'].str[0:7]
df_x_test['timestamp_8'] = df_x_test['timestamp'].str[0:8]
# 获取每个用户和每个电影的平均打分和评价数量
u_r = dict()
u_r_num = dict()
i_r = dict()
i_r_num = dict()
with open('../input/train.csv', 'r') as csvfile:
    fp = csv.reader(csvfile)
    for line in fp:
        userID, movieID, _, rate = line[:4]
        u_r.setdefault(userID, 0)
        u_r[userID] += int(rate)
        u_r_num.setdefault(userID, 0)
        u_r_num[userID] += 1
        i_r.setdefault(movieID, 0)
        i_r[movieID] += int(rate)
        i_r_num.setdefault(movieID, 0)
        i_r_num[movieID] += 1
# 943,1639 对比总用户数943,总电影数1682 发现有部分电影没有训练数据
# print(len(u_r), len(i_r))
for u in range(1, 944):
    u_r[str(u)] /= (1.0 * u_r_num[str(u)])
# 使用新字典,这样遍历后的新字典键值仍保持1-1682的升序
i_r_new = dict()
i_r_num_new = dict()
for i in range(1, 1683):
    if str(i) not in i_r.keys():
        i_r_new[str(i)] = 4.0
        i_r_num_new[str(i)] = 0
    else:
        i_r_new[str(i)] = i_r[str(i)] / (1.0 * i_r_num[str(i)])
        i_r_num_new[str(i)] = i_r_num[str(i)]
# print(len(u_r), len(i_r_new))
ur_value_list = list(u_r.values())
urn_value_list = list(u_r_num.values())
ir_new_value_list = list(i_r_new.values())
irn_new_value_list = list(i_r_num_new.values())
userInfo['average_u_rate'] = ur_value_list
itemInfo['average_i_rate'] = ir_new_value_list
userInfo['u_rate_num'] = urn_value_list
itemInfo['i_rated_num'] = irn_new_value_list
print(userInfo.head())
print(itemInfo.tail())

# 数据合并
df_x_train = pd.merge(df_x_train, userInfo, how='left', on='userId')
df_x_train = pd.merge(df_x_train, itemInfo, how='left', on='movie_id')
df_x_test = pd.merge(df_x_test, userInfo, how='left', on='userId')
df_x_test = pd.merge(df_x_test, itemInfo, how='left', on='movie_id')

# 删除不必要属性
df_y_train = df_x_train['rating']
df_x_train = df_x_train.drop(['release_date', 'video_release_date', 'IMDb_URL', 'Unnamed: 0',
                              'rating', 'movie_title'], axis=1)
df_x_test = df_x_test.drop(['release_date', 'video_release_date', 'IMDb_URL', 'Unnamed: 0',
                              'movie_title'], axis=1)
print(df_x_train.info())
# print(df_x_test.info())

# 类型属性
categorical_features_indices = np.where((df_x_train.dtypes != np.int64) & (df_x_train.dtypes != np.float64))[0]
print(categorical_features_indices)

# 五折交叉验证 ========================================================
# 创建分类/回归模型
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
oof = np.zeros(len(df_x_train))
predictions = np.zeros([len(df_x_test), 5])
for i, (trn, val) in enumerate(folds.split(df_x_train.values, df_y_train.values)):
    print("Fold:", i + 1)
    print("trn:", len(trn), "val:", len(val))

    trn_x = df_x_train.iloc[trn]
    trn_y = df_y_train.iloc[trn]
    val_x = df_x_train.iloc[val]
    val_y = df_y_train.iloc[val]

    clf = CatBoostClassifier(iterations=10000, depth=6, cat_features=categorical_features_indices, learning_rate=0.1,
                             loss_function='MultiClass', verbose=True,
                             task_type='GPU', devices='0-3', metric_period=50)
    clf.fit(
        trn_x, trn_y.astype('int32'),
        eval_set=(val_x, val_y.astype('int32')),
        early_stopping_rounds=200,
        verbose=True,)
        # use_best_model=True)
    oof[val] = clf.predict(df_x_train.iloc[val])[:, 0]
    predictions[:, i] = clf.predict(df_x_test)[:, 0]
# 五折交叉验证 ========================================================


# 模型评估 ========================================================
predicts = oof
hit = 0
for i in range(len(predicts)):
    if int(predicts[i]) == df_y_train.values[i]:
        hit += 1
print("evaluate {}, accuracy: {}".format(len(predicts), 1.0 * hit / len(predicts)))
# 模型评估 ========================================================


# 预测值 ========================================================
predicts = predictions
index = 0
with open("../input/test.csv", 'r', newline='') as readfile:
    with open("../sub/submit.csv", 'w', newline='') as writefile:
        csvreader = csv.reader(readfile)
        csvwriter = csv.writer(writefile)
        for line in csvreader:
            userID, movieID, rtime = line[:3]

            # bincount（）：统计非负整数的个数，不能统计浮点数
            counts = np.bincount(predicts[index].astype(np.int))
            # 返回众数
            rate = int(np.argmax(counts))

            row = [userID, movieID, rtime, rate]
            csvwriter.writerow(row)
            index += 1
            # print("No.{} evaluate: {}".format(index, rate))
# 预测值 ========================================================



# 特征重要性可视化
# fea_ = model.feature_importances_
# fea_name = model.feature_names_
# plt.figure(figsize=(10, 10))
# plt.barh(fea_name,fea_,height =0.5)
# plt.show()