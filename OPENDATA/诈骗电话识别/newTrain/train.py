# -*- coding: utf-8 -*-
# @Time     : 2020/6/13 17:30
# @Author   : Michael_Zhouy

import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE


def user_process(data_user, Train=1):
    data = data_user.copy()
    if Train:
        da = data.iloc[:, 4:-1]
        avg_data = da.mean(axis=1)
        avg_data.fillna(avg_data.mean(), inplace=True)
        data['avgfee'] = avg_data
        data['city_name'].fillna('无', inplace=True)
        data['county_name'].fillna('无', inplace=True)
        data = data.drop(['arpu_201908', 'arpu_201909', 'arpu_201910', 'arpu_201911',
                          'arpu_201912', 'arpu_202001', 'arpu_202002', 'arpu_202003'], axis=1)
    else:
        data['arpu_202004'] = data['arpu_202004'].fillna(data['arpu_202004'].mean())
        data['avgfee'] = data['arpu_202004']
        data['city_name'].fillna('无', inplace=True)
        data['county_name'].fillna('无', inplace=True)
        data = data.drop('arpu_202004', axis=1)

    return data


# lo_data = data_voc.iloc[:120000, :]


def voc_process(data_voc):
    lo_data = data_voc.copy()
    lo_data['start_datetime'] = pd.to_datetime(lo_data['start_datetime'], format='%Y/%m/%d %H:%M:%S')
    lo_data['month'] = lo_data['start_datetime'].dt.month

    """
    平均每月总的电话次数
    """
    lo_data['cnt'] = 1
    print(lo_data.columns)
    sum_of_all = pd.pivot_table(lo_data,
                                index=['phone_no_m', 'month'],
                                values='cnt',
                                aggfunc=sum).reset_index(level=['phone_no_m', 'month'])
    avg_sum_of_call = pd.pivot_table(sum_of_all, index=['phone_no_m'], values='cnt',
                                     aggfunc=np.mean).reset_index(level=['phone_no_m'])
    data = avg_sum_of_call.copy()

    """
    平均每月电话总时长（分钟）
    """
    print(lo_data.columns)
    sum_of_call_time = (pd.pivot_table(lo_data,
                                       index=['phone_no_m', 'month'],
                                       values='call_dur',
                                       aggfunc=sum) / 60).reset_index(level=['phone_no_m', 'month'])
    sum_of_call_time = pd.pivot_table(sum_of_call_time,
                                      index=['phone_no_m'],
                                      values='call_dur',
                                      aggfunc=np.mean).reset_index(level=['phone_no_m'])

    data = pd.merge(data, sum_of_call_time, on='phone_no_m', how='left')

    """
    统计主叫次数，被叫次数，传呼的次数以及占比
    """
    lo_data['calltype_id_1'] = (lo_data['calltype_id'] == 1).astype(int)
    lo_data['calltype_id_2'] = (lo_data['calltype_id'] == 2).astype(int)
    lo_data['calltype_id_3'] = (lo_data['calltype_id'] == 3).astype(int)

    sum_of_call_calltype_id_1 = pd.pivot_table(lo_data,
                                               index=['phone_no_m', 'month'],
                                               values='calltype_id_1',
                                               aggfunc=sum).reset_index(level=['phone_no_m', 'month'])
    sum_of_call_calltype_id_2 = pd.pivot_table(lo_data,
                                               index=['phone_no_m', 'month'],
                                               values='calltype_id_2',
                                               aggfunc=sum).reset_index(level=['phone_no_m', 'month'])
    sum_of_call_calltype_id_3 = pd.pivot_table(lo_data,
                                               index=['phone_no_m', 'month'],
                                               values='calltype_id_3',
                                               aggfunc=sum).reset_index(level=['phone_no_m', 'month'])

    sum_of_call_calltype_id_1 = pd.pivot_table(sum_of_call_calltype_id_1,
                                               index=['phone_no_m'],
                                               values='calltype_id_1',
                                               aggfunc=np.mean).reset_index(level=['phone_no_m'])
    sum_of_call_calltype_id_2 = pd.pivot_table(sum_of_call_calltype_id_2,
                                               index=['phone_no_m'],
                                               values='calltype_id_2',
                                               aggfunc=np.mean).reset_index(level=['phone_no_m'])
    sum_of_call_calltype_id_3 = pd.pivot_table(sum_of_call_calltype_id_3,
                                               index=['phone_no_m'],
                                               values='calltype_id_3',
                                               aggfunc=np.mean).reset_index(level=['phone_no_m'])

    data = pd.merge(data, sum_of_call_calltype_id_1, on='phone_no_m', how='left')
    data = pd.merge(data, sum_of_call_calltype_id_2, on='phone_no_m', how='left')
    data = pd.merge(data, sum_of_call_calltype_id_3, on='phone_no_m', how='left')

    # 占比
    data['sum_of_call_calltype_id'] = data['calltype_id_1'] + data['calltype_id_2'] + data['calltype_id_3']
    data['calltype_id_1_rate'] = data['calltype_id_1'] / data['sum_of_call_calltype_id']
    data['calltype_id_2_rate'] = data['calltype_id_2'] / data['sum_of_call_calltype_id']
    data['calltype_id_3_rate'] = data['calltype_id_3'] / data['sum_of_call_calltype_id']

    """
    对多少个不同的号码打了电话
    """
    lo_data['sum_of_different_call'] = 1
    sum_of_different_call = pd.pivot_table(lo_data,
                                           index=['phone_no_m', 'opposite_no_m'],
                                           values='sum_of_different_call',
                                           aggfunc=sum).reset_index(level=['phone_no_m', 'opposite_no_m'])
    sum_of_different_call['sum_of_different_call'] = 1
    sum_of_different_call = pd.pivot_table(sum_of_different_call,
                                           index=['phone_no_m'],
                                           values='sum_of_different_call',
                                           aggfunc=sum).reset_index(level=['phone_no_m'])

    """
    对多少个不同的IMEI设备打过电话
    """
    lo_data['sum_of_different_imei'] = 1
    sum_of_different_imei = pd.pivot_table(lo_data,
                                           index=['phone_no_m', 'imei_m'],
                                           values='sum_of_different_imei',
                                           aggfunc=sum).reset_index(level=['phone_no_m', 'imei_m'])
    sum_of_different_imei['sum_of_different_imei'] = 1
    sum_of_different_imei = pd.pivot_table(sum_of_different_imei,
                                           index=['phone_no_m'],
                                           values='sum_of_different_imei',
                                           aggfunc=sum).reset_index(level=['phone_no_m'])

    """"
    平均通话时长
    """


    """
    平均每月通话次数
    """


    """
    平均每月时长
    """

    """
    平均每月 主叫 / 被叫 通话时长
    """

    """
    异地被叫/呼叫次数
    """

    """
    每日平均时间段打电话次数
    """

    return data


# lo_data = data_sms.iloc[:120000, :]

def sms_process(data_sms):

    lo_data = data_sms.copy()

    """
    时间处理
    """
    lo_data['request_datetime'] = pd.to_datetime(lo_data['request_datetime'],
                                                 format='%Y/%m/%d %H:%M:%S')
    lo_data['month'] = lo_data['request_datetime'].dt.month

    """
    总的短信次数
    """
    lo_data['sum_of_sms'] = 1
    print(lo_data.columns)
    sum_of_sms = pd.pivot_table(lo_data,
                                index=['phone_no_m', 'month'],
                                values='sum_of_sms',
                                aggfunc=sum).reset_index(level=['phone_no_m', 'month'])
    sum_of_call_time = pd.pivot_table(sum_of_sms,
                                      index=['phone_no_m'],
                                      values='sum_of_sms',
                                      aggfunc=np.mean).reset_index(level=['phone_no_m'])
    data = sum_of_call_time.copy()


    """
    平均每月发的短信条数
    """

    """
    平均每月给多少个人发过短信
    """

    """
    平均每天发短信条数
    """

    """
    平均每月发短信的天数
    """

    """
    时间段按照编码，统计每天时间段发短信的条数 0-24H
    """

    """
    出现5s内发送多条短信的次数
    """

    """
    统计上行，下行次数及其占比
    """
    lo_data['smstype_id_1'] = (lo_data['calltype_id'] == 1).astype(int)
    lo_data['smstype_id_2'] = (lo_data['calltype_id'] == 2).astype(int)

    sum_of_call_calltype_id_1 = pd.pivot_table(lo_data,
                                               index=['phone_no_m'],
                                               values='smstype_id_1',
                                               aggfunc=sum).reset_index(level=['phone_no_m'])
    sum_of_call_calltype_id_2 = pd.pivot_table(lo_data,
                                               index=['phone_no_m'],
                                               values='smstype_id_2',
                                               aggfunc=sum).reset_index(level=['phone_no_m'])
    data = pd.merge(data, sum_of_call_calltype_id_1, on='phone_no_m', how='left')
    data = pd.merge(data, sum_of_call_calltype_id_2, on='phone_no_m', how='left')

    # 占比
    data['sum_of_sms_smstype_id'] = data['smstype_id_1'] + data['smstype_id_2']
    data['smstype_id_1_rate'] = data['smstype_id_1'] / data['sum_of_sms_smstype_id']
    data['smstype_id_2_rate'] = data['smstype_id_2'] / data['sum_of_sms_smstype_id']

    """
    对多少个不同的号码发送给短信
    """
    lo_data['cnt'] = 1
    sum_of_different_sms = pd.pivot_table(lo_data,
                                          index=['phone_no_m', 'opposite_no_m'],
                                          values='cnt',
                                          aggfunc=sum).reset_index(level=['phone_no_m'])
    sum_of_different_sms['sum_of_different_sms'] = 1
    sum_of_different_sms = pd.pivot_table(sum_of_different_sms,
                                          index=['phone_no_m'],
                                          values='sum_of_different_sms',
                                          aggfunc=sum).reset_index(level=['phone_no_m'])
    data = pd.merge(data, sum_of_different_sms, on='phone_no_m', how='left')


    return data


# lo_data = data_app.iloc[:120000, :]

def app_process(data_app):
    lo_data = data_app.copy()

    lo_data['busi_name'].fillna('其他', inplace=True)
    lo_data = lo_data.fillna(method='pad', axis=0)
    lo_data = lo_data.fillna(method='backfill', axis=0)

    """
    总的流量
    """
    # lo_data.columns
    # sum_of_flow = pd.pivot_table(lo_data,
    #                              index=['phone_no_m'],
    #                              values='flow',
    #                              aggfunc=sum).reset_index(level=['phone_no_m'])
    # sum_of_flow.columns = ['phone_no_m', 'sum_of_flow']
    # data = sum_of_flow.copy()

    """
    每月平均流量
    """
    avg_of_flow = pd.pivot_table(lo_data,
                                 index=['phone_no_m', 'month_id'],
                                 values='flow',
                                 aggfunc=sum).reset_index(level=['phone_no_m', 'month_id'])
    avg_of_flow = pd.pivot_table(avg_of_flow,
                                 index=['phone_no_m'],
                                 values='flow',
                                 aggfunc=np.mean).reset_index(level=['phone_no_m'])
    data = avg_of_flow

    """
    使用了多少个APP
    """
    lo_data['cnt'] = 1
    sum_of_app = pd.pivot_table(lo_data,
                                index=['phone_no_m', 'busi_name'],
                                values='cnt',
                                aggfunc=sum).reset_index(level=['phone_no_m', 'busi_name'])
    sum_of_app['sum_of_app'] = 1
    sum_of_app = pd.pivot_table(sum_of_app,
                                index=['phone_no_m'],
                                values='sum_of_app',
                                aggfunc=sum).reset_index(level=['phone_no_m'])
    data = pd.merge(data, sum_of_app, on='phone_no_m', how='left')


    """
    平均每个APP使用的流量
    """
    data['app_use_of_flow_avg'] = data['flow'] / data['sum_of_app']

    """
    编码  流量用的最多的月份，用的最少的月份
    """

    return data


def data_concat(data_app, data_sms, data_voc, data_user, train=1):
    print(data_user.columns)
    if train:
        user = user_process(data_user, Train=1)
        data = user[['phone_no_m', 'idcard_cnt', 'avgfee', 'label']]
        # data = user[['phone_no_m', 'idcart_cnt']]
    else:
        user = user_process(data_user, Train=0)
        data = user[['phone_no_m', 'idcard_cnt', 'avgfee']]
        # data = user[['phone_no_m', 'idcard_cnt']]


    voc = voc_process(data_voc)
    sms = sms_process(data_sms)
    app = app_process(data_app)
    data = pd.merge(data, voc, on='phone_no_m', how='left')
    data = pd.merge(data, sms, on='phone_no_m', how='left')
    data = pd.merge(data, app, on='phone_no_m', how='left')

    if train:
        data['label'] = user['label']
    return data


def f1_loss(y, pred):
    beta = 2
    p = 1. / (1. + np.exp(-pred))
    grad = p * ((beta - 1) * y + 1) - beta * y
    hess = ((beta - 1) * y + 1) * p * (1.0 - p)
    return grad, hess


def train(data_notnull, test):
    print(data_notnull.columns)
    label = data_notnull['label']
    data = data_notnull.drop(['label', 'phone_no_m'], axis=1)

    oversampler = SMOTE(random_state=0)
    data, label = oversampler.fit_resample(data, label)

    len(label[label == 1])

    data = pd.DataFrame(data)
    label = pd.Series(label)

    x_train, x_test, y_train, y_test = train_test_split(data, label,
                                                        test_size=0.1,
                                                        random_state=2020)

    params = {'num_leaves': 60,  # 结果对最终效果影响较大，越大值越好，太大会出现过拟合
              'min_data_in_leaf': 30,
              'objective': 'binary',  # 定义的目标函数
              'max_depth': 4,
              'learning_rate': 0.01,
              'min_sum_hessian_in_leaf': 6,
              'boosting': 'gbdt',
              'feature_fraction': 0.9,  # 提取的特征比率
              'bagging_freq': 1,
              'bagging_fraction': 0.8,
              'bagging_seed': 11,
              'lambda_l1': 0.2,  # l1正则
              # 'lambda_l2': 0.05,     #l2正则
              'verbosity': -1,
              'nthread': -1,  # 线程数量，-1表示全部线程，线程越多，运行的速度越快
              'metric': {'binary_logloss', 'auc'},  ##评价函数选择
              'random_state': 2020,  # 随机数种子，可以防止每次运行的结果不一致

              # 'device': 'gpu'  # 如果安装的事gpu版本的lightgbm,可以加快运算
              }

    trn_data = lgb.Dataset(x_train, label=y_train)
    val_data = lgb.Dataset(x_test, label=y_test, reference=trn_data)

    # clf = lgb.LGBMClassifier(params =params)
    # clf.set_params(**{"objective": f1_loss})
    # clf.fit(x_train, y_train,
    #         eval_set=[(x_test, y_test)],
    #         eval_metric=lambda y_true, y_pred: [custom_f1_eval(y_true, y_pred)],
    #         early_stopping_rounds=50,
    #         verbose=10)

    clf = lgb.train(params,
                    trn_data,
                    num_boost_round=5000,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=5,
                    early_stopping_rounds=100)
    predict = clf.predict(x_test, num_iteration=clf.best_iteration)
    predict = predict >= 0.5
    score = f1_score(predict, y_test, average='macro')
    print('*' * 10)
    print('score: ', score)

    """
    预测
    """
    sub = pd.read_csv('../sub/submit_example.csv')
    test = test.drop(['phone_no_m'], axis=1)
    predict_test = clf.predict(test, num_iteration=clf.best_iteration)
    sub['label'] = (predict_test >= 0.5).astype(int)
    sub.to_csv('../sub/sub_{}.csv'.format(time.strftime('%Y%m%d')), index=False)


if __name__ == '__main__':
    data_app = pd.read_csv('../input/train/train_app.csv')
    data_sms = pd.read_csv('../input/train/train_sms.csv')
    data_voc = pd.read_csv('../input/train/train_voc.csv')
    data_user = pd.read_csv('../input/train/train_user.csv')

    test_app = pd.read_csv('../input/test/test_app.csv')
    test_sms = pd.read_csv('../input/test/test_sms.csv')
    test_voc = pd.read_csv('../input/test/test_voc.csv')
    test_user = pd.read_csv('../input/test/test_user.csv')

    data = data_concat(data_app, data_sms, data_voc, data_user, train=1)
    # 参看1，生成字典，key为列名，value为列对应的均值
    values = dict([(col_name, col_mean) for col_name, col_mean in zip(data.columns.tolist(), data.mean().tolist())])
    data.fillna(value=values, inplace=True)
    data_notnull = data

    test = data_concat(test_app, test_sms, test_voc, test_user, train=0)
    # 参看1，生成字典，key为列名，value为列对应的均值
    values = dict([(col_name, col_mean) for col_name, col_mean in zip(test.columns.tolist(), test.mean().tolist())])
    test.fillna(value=values, inplace=True)

    train(data_notnull, test)
