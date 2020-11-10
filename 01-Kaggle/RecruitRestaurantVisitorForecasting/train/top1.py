# -*- coding: utf-8 -*-
# @Time     : 2020/7/16 14:37
# @Author   : Michael_Zhouy

import time
import numpy as np
import pandas as pd
from dateutil.parser import parse
from datetime import date, timedelta
from sklearn.preprocessing import LabelEncoder
import datetime
import lightgbm as lgb

data_path = '../input/'

air_reserve = pd.read_csv(data_path + 'air_reserve.csv').rename(columns={'air_store_id': 'store_id'})
hpg_reserve = pd.read_csv(data_path + 'hpg_reserve.csv').rename(columns={'hpg_store_id': 'store_id'})
air_store = pd.read_csv(data_path + 'air_store_info.csv').rename(columns={'air_store_id': 'store_id'})
hpg_store = pd.read_csv(data_path + 'hpg_store_info.csv').rename(columns={'hpg_store_id': 'store_id'})
air_visit = pd.read_csv(data_path + 'air_visit_data.csv').rename(columns={'air_store_id': 'store_id'})
store_id_map = pd.read_csv(data_path + 'store_id_relation.csv').set_index('hpg_store_id', drop=False)
date_info = pd.read_csv(data_path + 'date_info.csv').rename(columns={'calendar_date': 'visit_date'}).drop('day_of_week',
                                                                                                          axis=1)
submission = pd.read_csv(data_path + 'sample_submission.csv')

# 提取日期，store_id
submission['visit_date'] = submission['id'].str[-10:]
submission['store_id'] = submission['id'].str[:-11]

# air预定表
air_reserve['visit_date'] = air_reserve['visit_datetime'].str[:10]  # 入店日期
air_reserve['reserve_date'] = air_reserve['reserve_datetime'].str[:10]  # 预定日期
air_reserve['dow'] = pd.to_datetime(air_reserve['visit_date']).dt.dayofweek  # 入店日期是周几，0-周一，6-周日

# hpg预定表
hpg_reserve['visit_date'] = hpg_reserve['visit_datetime'].str[:10]
hpg_reserve['reserve_date'] = hpg_reserve['reserve_datetime'].str[:10]
hpg_reserve['dow'] = pd.to_datetime(hpg_reserve['visit_date']).dt.dayofweek

# 主表，唯一id
air_visit['id'] = air_visit['store_id'] + '_' + air_visit['visit_date']

# 映射
hpg_reserve['store_id'] = hpg_reserve['store_id'].map(store_id_map['air_store_id']).fillna(hpg_reserve['store_id'])
hpg_store['store_id'] = hpg_store['store_id'].map(store_id_map['air_store_id']).fillna(hpg_store['store_id'])
hpg_store.rename(columns={'hpg_genre_name': 'air_genre_name',
                          'hpg_area_name': 'air_area_name'},
                 inplace=True)

# 主表
data = pd.concat([air_visit, submission]).copy()

# 主表visit_date是周几
data['dow'] = pd.to_datetime(data['visit_date']).dt.dayofweek

# 日期表，日期是否是周末或者节假日
date_info['holiday_flg2'] = pd.to_datetime(date_info['visit_date']).dt.dayofweek  # 周几
date_info['holiday_flg2'] = ((date_info['holiday_flg2'] > 4) | (date_info['holiday_flg'] == 1)).astype(int)

# air_store_info表，取第一项
air_store['air_area_name0'] = air_store['air_area_name'].apply(lambda x: x.split(' ')[0])
lbl = LabelEncoder()
air_store['air_genre_name'] = lbl.fit_transform(air_store['air_genre_name'])
air_store['air_area_name0'] = lbl.fit_transform(air_store['air_area_name0'])

# 将入店人数做log变换
data['visitors'] = np.log1p(data['visitors'])
# 关联air_store表
data = data.merge(air_store,
                  on='store_id',
                  how='left')
# 关联date_info表
data = data.merge(date_info[['visit_date', 'holiday_flg', 'holiday_flg2']],
                  on=['visit_date'],
                  how='left')


###################################### function #########################################
def concat(L):
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            try:
                result[l.columns.tolist()] = l
            except:
                print(l.head())
    return result


def left_merge(data1, data2, on):
    """
    两个表做左关联
    @param data1: 左表
    @param data2: 右表
    @param on: 关联键
    @return:
    """
    # 判断关联键的类型是否为list，不是的话将其转换为list
    if type(on) != list:
        on = [on]
    # 判断右表data2的列名是否包含了关联键on，没有的话，将右表data2的索引设置成列
    if (set(on) & set(data2.columns)) != set(on):
        data2_temp = data2.reset_index()
    else:
        data2_temp = data2.copy()
    # 右表data2的列中，不包含on的列
    columns = [f for f in data2.columns if f not in on]
    result = data1.merge(data2_temp, on=on, how='left')
    result = result[columns]
    return result


def diff_of_days(day1, day2):
    """
    计算两个日期相差的天数，day1-day2
    @param day1:
    @param day2:
    @return:
    """
    days = (parse(day1[:10]) - parse(day2[:10])).days
    return days


def date_add_days(start_date, days):
    """
    start_date + days
    @param start_date:
    @param days:
    @return:
    """
    end_date = parse(start_date[:10]) + timedelta(days=days)
    end_date = end_date.strftime('%Y-%m-%d')
    return end_date


def get_label(end_date, n_day):
    """
    关键操作
    """
    label_end_date = date_add_days(end_date, n_day)  # 日期加天数
    label = data[(data['visit_date'] < label_end_date) & (data['visit_date'] >= end_date)].copy()  # 取两个日期之间的数据
    label['end_date'] = end_date

    # 计算visit_date和end_date之间相差的天数
    label['diff_of_day'] = label['visit_date'].apply(lambda x: diff_of_days(x, end_date))
    label['month'] = label['visit_date'].str[5:7].astype(int)  # visit_date所在的月份
    label['year'] = label['visit_date'].str[:4].astype(int)  # visit_date所在的年份
    # date_info表中日期加天数
    for i in [3, 2, 1, -1]:
        date_info_temp = date_info.copy()
        date_info_temp['visit_date'] = date_info_temp['visit_date'].apply(lambda x: date_add_days(x, i))
        # visit_date的前几天或后几天是否是节假日
        date_info_temp.rename(columns={'holiday_flg': 'ahead_holiday_{}'.format(i),
                                       'holiday_flg2': 'ahead_holiday2_{}'.format(i)},
                              inplace=True)
        label = label.merge(date_info_temp, on=['visit_date'], how='left')
    label = label.reset_index(drop=True)
    return label


def get_store_visitor_feat(label, key, n_day):
    """
    get_store_visitor_feat(label, key, 1000)
    时间序列，入店的前多少天，店铺的统计量
    @param label:
    @param key:
    @param n_day:
    @return:
    """
    start_date = date_add_days(key[0], -n_day)  # 前几天
    data_temp = data[(data.visit_date < key[0]) & (data.visit_date > start_date)].copy()
    result = data_temp.groupby(['store_id'], as_index=False)['visitors'].agg({
        'store_min{}'.format(n_day): 'min',
        'store_mean{}'.format(n_day): 'mean',
        'store_median{}'.format(n_day): 'median',
        'store_max{}'.format(n_day): 'max',
        'store_count{}'.format(n_day): 'count',
        'store_std{}'.format(n_day): 'std',
        'store_skew{}'.format(n_day): 'skew'
    })
    result = left_merge(label, result, on=['store_id']).fillna(0)
    return result


def get_store_exp_visitor_feat(label, key, n_day):
    """
    key[0] 传入一个日期，减几天，得到一个日期
    入店日期在这两个日期之间的所有入店信息
    visit_date: 日期差，计算入店日期与key[0] 相差的天数
    weight: 0.985 ** visit_date
    visitors: visitors * weight
    result1: 在两个日期之间，加权visitors求和
    result2: 在两个日期之间，加权天数weight求和
    store_exp_mean: 加权平均每天的visitors
    @param label:
    @param key:
    @param n_day:
    @return:
    """
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (data.visit_date > start_date)].copy()
    data_temp['visit_date'] = data_temp['visit_date'].apply(lambda x: diff_of_days(key[0], x))  # 两个日期相差的天数
    data_temp['weight'] = data_temp['visit_date'].apply(lambda x: 0.985 ** x)  # 0.985 ** 相差天数

    data_temp['visitors'] = data_temp['visitors'] * data_temp['weight']  # 客户数 * weight

    # 总客户数
    result1 = data_temp.groupby(['store_id'], as_index=False)['visitors'].agg({
        'store_exp_mean{}'.format(n_day): 'sum'
    })
    # 总天数
    result2 = data_temp.groupby(['store_id'], as_index=False)['weight'].agg(
        {'store_exp_weight_sum{}'.format(n_day): 'sum'})
    result = result1.merge(result2, on=['store_id'], how='left')
    # 平均每天的客户数
    result['store_exp_mean{}'.format(n_day)] = result['store_exp_mean{}'.format(n_day)] / result[
        'store_exp_weight_sum{}'.format(n_day)]
    result = left_merge(label, result, on=['store_id']).fillna(0)
    return result


def get_store_week_feat(label, key, n_day):
    """
    时间序列，入店的前多少天
    每个店铺，每周几客户数的统计量
    groupby ['store_id', 'dow']
    @param label:
    @param key:
    @param n_day:
    @return:
    """
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (data.visit_date > start_date)].copy()
    result = data_temp.groupby(['store_id', 'dow'], as_index=False)['visitors'].agg({
        'store_dow_min{}'.format(n_day): 'min',
        'store_dow_mean{}'.format(n_day): 'mean',
        'store_dow_median{}'.format(n_day): 'median',
        'store_dow_max{}'.format(n_day): 'max',
        'store_dow_count{}'.format(n_day): 'count',
        'store_dow_std{}'.format(n_day): 'std',
        'store_dow_skew{}'.format(n_day): 'skew'
    })
    result = left_merge(label, result, on=['store_id', 'dow']).fillna(0)
    return result


def get_store_week_diff_feat(label, key, n_day):
    """
    时间序列，入店的前多少天
    每个店铺，每天客户数差的统计量
    groupby ['store_id', 'visit_date']
    @param label:
    @param key:
    @param n_day:
    @return:
    """
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (data.visit_date > start_date)].copy()
    """
    data_temp = data_temp[['store_id', 'visit_date', 'visit_date']]
    data_old_tmp = data_temp.copy()
    data_old_tmp['visit_date'] = data_temp['visit_date'].apply(lambda x: x + timedelta(1)).values  # 用昨天的日期去关联
    data_old_tmp.rename(columns={'visitors': 'visitors_yesterday'},
                        inplace=True)
    data_temp = data_temp.merge(data_old_tmp[['store_id', 'visit_date', 'visitors_yesterday']],
                                on=['store_id', 'visit_date'],
                                how='left')
    data_temp['visitors_diff'] = data_temp['visitors'] - data_temp['visitors_yesterday']
    data_temp = data_temp.loc[data_temp['visitors_diff'].notnull()]
    """
    # unstack() 将内层行索引变成列索引
    # 即结果为result的行索引为stored_id，列索引为visit_date，取值为visitors
    result = data_temp.set_index(['store_id', 'visit_date'])['visitors'].unstack()
    # diff() 后一项减前一项，并过滤掉第一列
    result = result.diff(axis=1).iloc[:, 1:]
    c = result.columns  # c = 'visit_date'的取值
    result['store_diff_mean'] = np.abs(result[c]).mean(axis=1)  # 每天增减人数的均值
    result['store_diff_std'] = result[c].std(axis=1)
    result['store_diff_max'] = result[c].max(axis=1)
    result['store_diff_min'] = result[c].min(axis=1)
    result = left_merge(label, result[['store_diff_mean', 'store_diff_std', 'store_diff_max', 'store_diff_min']],
                        on=['store_id']).fillna(0)
    return result


def get_store_all_week_feat(label, key, n_day):
    """
    时间序列，入店的前多少天，周几的统计量
    周一(0) 客户数的统计量
    周二(1) 客户数的统计量
    @param label:
    @param key:
    @param n_day:
    @return:
    """
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (data.visit_date > start_date)].copy()
    result_temp = data_temp.groupby(['store_id', 'dow'], as_index=False)['visitors'].agg({
        'store_dow_mean{}'.format(n_day): 'mean',
        'store_dow_median{}'.format(n_day): 'median',
        'store_dow_sum{}'.format(n_day): 'max',
        'store_dow_count{}'.format(n_day): 'count'
    })
    result = pd.DataFrame()
    for i in range(7):
        result_sub = result_temp[result_temp['dow'] == i].copy()
        result_sub = result_sub.set_index('store_id')
        result_sub = result_sub.add_prefix(str(i))  # 列名后加str(i)
        result_sub = left_merge(label, result_sub, on=['store_id']).fillna(0)
        result = pd.concat([result, result_sub],
                           axis=1)
    return result


def get_store_week_exp_feat(label, key, n_day):
    """
    时间序列，入店前的多少天，加权每周几的总客户数，加权平均每天客户数
    @param label:
    @param key:
    @param n_day:
    @return:
    """
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (data.visit_date > start_date)].copy()
    # 两个日期相差的天数
    data_temp['visit_date'] = data_temp['visit_date'].apply(lambda x: diff_of_days(key[0], x))
    data_temp['visitors2'] = data_temp['visitors']
    result = None
    for i in [0.9, 0.95, 0.97, 0.98, 0.985, 0.99, 0.999, 0.9999]:
        data_temp['weight'] = data_temp['visit_date'].apply(lambda x: i ** x)  # 两个日期相差的天数的权重
        data_temp['visitors1'] = data_temp['visitors'] * data_temp['weight']  # 客户数*权重
        data_temp['visitors2'] = data_temp['visitors2'] * data_temp['weight']
        result1 = data_temp.groupby(['store_id', 'dow'], as_index=False)['visitors1'].agg({
            'store_dow_exp_mean{}_{}'.format(n_day, i): 'sum'
        })
        result3 = data_temp.groupby(['store_id', 'dow'], as_index=False)['visitors2'].agg({
            'store_dow_exp_mean2{}_{}'.format(n_day, i): 'sum'
        })
        result2 = data_temp.groupby(['store_id', 'dow'], as_index=False)['weight'].agg({
            'store_dow_exp_weight_sum{}_{}'.format(n_day, i): 'sum'
        })
        result_temp = result1.merge(result2,
                                    on=['store_id', 'dow'],
                                    how='left')
        result_temp = result_temp.merge(result3,
                                        on=['store_id', 'dow'],
                                        how='left')
        result_temp['store_dow_exp_mean{}_{}'.format(n_day, i)] = (
                    result_temp['store_dow_exp_mean{}_{}'.format(n_day, i)]
                    / result_temp['store_dow_exp_weight_sum{}_{}'.format(n_day, i)])
        result_temp['store_dow_exp_mean2{}_{}'.format(n_day, i)] = (
                    result_temp['store_dow_exp_mean2{}_{}'.format(n_day, i)]
                    / result_temp['store_dow_exp_weight_sum{}_{}'.format(n_day, i)])
        if result is None:
            result = result_temp
        else:
            result = result.merge(result_temp,
                                  on=['store_id', 'dow'],
                                  how='left')
    result = left_merge(label, result, on=['store_id', 'dow']).fillna(0)
    return result


def get_store_holiday_feat(label, key, n_day):
    """
    时间序列，入店的前多少天
    店铺在节假日的客户数的统计量，groupby(['store_id', 'holiday_flg'], as_index=False)
    店铺在节假日或周末的客户数的统计量，groupby(['store_id', 'holiday_flg2'], as_index=False)
    @param label:
    @param key:
    @param n_day:
    @return:
    """
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (data.visit_date > start_date)].copy()
    # holiday_flg: 节假日
    result1 = data_temp.groupby(['store_id', 'holiday_flg'], as_index=False)['visitors'].agg({
        'store_holiday_min{}'.format(n_day): 'min',
        'store_holiday_mean{}'.format(n_day): 'mean',
        'store_holiday_median{}'.format(n_day): 'median',
        'store_holiday_max{}'.format(n_day): 'max',
        'store_holiday_count{}'.format(n_day): 'count',
        'store_holiday_std{}'.format(n_day): 'std',
        'store_holiday_skew{}'.format(n_day): 'skew'
    })
    result1 = left_merge(label, result1, on=['store_id', 'holiday_flg']).fillna(0)

    # holiday_flg2: 节假日+周末
    result2 = data_temp.groupby(['store_id', 'holiday_flg2'], as_index=False)['visitors'].agg({
        'store_holiday2_min{}'.format(n_day): 'min',
        'store_holiday2_mean{}'.format(n_day): 'mean',
        'store_holiday2_median{}'.format(n_day): 'median',
        'store_holiday2_max{}'.format(n_day): 'max',
        'store_holiday2_count{}'.format(n_day): 'count',
        'store_holiday2_std{}'.format(n_day): 'std',
        'store_holiday2_skew{}'.format(n_day): 'skew'
    })
    result2 = left_merge(label, result2, on=['store_id', 'holiday_flg2']).fillna(0)

    result = pd.concat([result1, result2], axis=1)
    return result


def get_genre_visitor_feat(label, key, n_day):
    """
    时间序列，入店的前多少天
    不同类型店铺的客户数的统计量，groupby(['air_genre_name'], as_index=False)
    @param label:
    @param key:
    @param n_day:
    @return:
    """
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (data.visit_date > start_date)].copy()
    result = data_temp.groupby(['air_genre_name'], as_index=False)['visitors'].agg({
        'genre_min{}'.format(n_day): 'min',
        'genre_mean{}'.format(n_day): 'mean',
        'genre_median{}'.format(n_day): 'median',
        'genre_max{}'.format(n_day): 'max',
        'genre_count{}'.format(n_day): 'count',
        'genre_std{}'.format(n_day): 'std',
        'genre_skew{}'.format(n_day): 'skew'
    })
    result = left_merge(label, result, on=['air_genre_name']).fillna(0)
    return result


def get_genre_exp_visitor_feat(label, key, n_day):
    """
    时间序列，入店的前多少天
    不同类型店铺的加权客户数的统计量，groupby(['air_genre_name'], as_index=False)
    @param label:
    @param key:
    @param n_day:
    @return:
    """
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (data.visit_date > start_date)].copy()
    data_temp['visit_date'] = data_temp['visit_date'].apply(lambda x: diff_of_days(key[0], x))  # 两个日期相差的天数
    data_temp['weight'] = data_temp['visit_date'].apply(lambda x: 0.985 ** x)  # 两个日期相差的天数的权重
    data_temp['visitors'] = data_temp['visitors'] * data_temp['weight']  # 客户数*权重
    # 不同类型店铺的加权客户数的求和
    result1 = data_temp.groupby(['air_genre_name'], as_index=False)['visitors'].agg({
        'genre_exp_mean{}'.format(n_day): 'sum'
    })
    # 不同类型店铺的加权天数的求和
    result2 = data_temp.groupby(['air_genre_name'], as_index=False)['weight'].agg({
        'genre_exp_weight_sum{}'.format(n_day): 'sum'
    })
    result = result1.merge(result2,
                           on=['air_genre_name'],
                           how='left')
    # 不同类型店铺的每天加权平均客户数
    result['genre_exp_mean{}'.format(n_day)] = (result['genre_exp_mean{}'.format(n_day)]
                                                / result['genre_exp_weight_sum{}'.format(n_day)])
    result = left_merge(label, result, on=['air_genre_name']).fillna(0)
    return result


def get_genre_week_feat(label, key, n_day):
    """
    时间序列，入店的前多少天
    不同类型店铺每周几的客户数的统计量，groupby(['air_genre_name', 'dow'], as_index=False)
    @param label:
    @param key:
    @param n_day:
    @return:
    """
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (data.visit_date > start_date)].copy()
    result = data_temp.groupby(['air_genre_name', 'dow'], as_index=False)['visitors'].agg({
        'genre_dow_min{}'.format(n_day): 'min',
        'genre_dow_mean{}'.format(n_day): 'mean',
        'genre_dow_median{}'.format(n_day): 'median',
        'genre_dow_max{}'.format(n_day): 'max',
        'genre_dow_count{}'.format(n_day): 'count',
        'genre_dow_std{}'.format(n_day): 'std',
        'genre_dow_skew{}'.format(n_day): 'skew'
    })
    result = left_merge(label, result, on=['air_genre_name', 'dow']).fillna(0)
    return result


def get_genre_week_exp_feat(label, key, n_day):
    """
    时间序列，入店的前多少天
    不同类型店铺每周几的加权客户数的统计量，groupby(['air_genre_name', 'dow'], as_index=False)
    @param label:
    @param key:
    @param n_day:
    @return:
    """
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (data.visit_date > start_date)].copy()
    data_temp['visit_date'] = data_temp['visit_date'].apply(lambda x: diff_of_days(key[0], x))  # 两个日期相差的天数
    data_temp['weight'] = data_temp['visit_date'].apply(lambda x: 0.985 ** x)
    data_temp['visitors'] = data_temp['visitors'] * data_temp['weight']
    result1 = data_temp.groupby(['air_genre_name', 'dow'], as_index=False)['visitors'].agg({
        'genre_dow_exp_mean{}'.format(n_day): 'sum'
    })
    result2 = data_temp.groupby(['air_genre_name', 'dow'], as_index=False)['weight'].agg({
        'genre_dow_exp_weight_sum{}'.format(n_day): 'sum'
    })
    result = result1.merge(result2,
                           on=['air_genre_name', 'dow'],
                           how='left')
    result['genre_dow_exp_mean{}'.format(n_day)] = (result['genre_dow_exp_mean{}'.format(n_day)]
                                                    / result['genre_dow_exp_weight_sum{}'.format(n_day)])
    result = left_merge(label, result, on=['air_genre_name', 'dow']).fillna(0)
    return result


def get_first_last_time(label, key, n_day):
    """
    时间序列，入店的前多少天
    每个店铺，每个入店日期距开业日期相隔的天数
    每个店铺，每个入店日期距最后一天相隔的天数
    @param label:
    @param key:
    @param n_day:
    @return:
    """
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (data.visit_date > start_date)].copy()

    # 根据入店日期进行排序
    data_temp = data_temp.sort_values('visit_date')
    # 每个入店日期距开业日期相隔的天数，每个入店日期距最后一天相隔的天数
    result = data_temp.groupby('store_id')['visit_date'].agg(first_time=lambda x: diff_of_days(key[0], np.min(x)),
                                                             last_time=lambda x: diff_of_days(key[0], np.max(x)))
    result = left_merge(label, result, on=['store_id']).fillna(0)
    return result


# air_reserve
def get_reserve_feat(label, key):
    """
    key[0] 是'2017-04-23'
    @param label:
    @param key:
    @return:
    """
    label_end_date = date_add_days(key[0], key[1])
    # air预定表的时间窗口
    air_reserve_temp = air_reserve[(air_reserve.visit_date >= key[0]) &  # key[0] 是'2017-04-23'
                                   (air_reserve.visit_date < label_end_date) &  # label_end_date 是'2017-05-31'
                                   (air_reserve.reserve_date < key[0])].copy()  # 预定日期在入店日期之前
    air_reserve_temp = air_reserve_temp.merge(air_store,
                                              on='store_id',
                                              how='left')

    # 预定的入店日期与预定日期相隔的天数
    air_reserve_temp['diff_time'] = (pd.to_datetime(air_reserve['visit_datetime'])
                                     - pd.to_datetime(air_reserve['reserve_datetime'])).dt.days
    air_reserve_temp = air_reserve_temp.merge(air_store, on='store_id')

    # air预定表，每个店铺，每天预定的总客户数和预定次数
    air_result = air_reserve_temp.groupby(['store_id', 'visit_date'])['reserve_visitors'].agg(
        air_reserve_visitors='sum',
        air_reserve_count='count'
    )

    # air预定表，每个store_id，每天预定的入店日期与预定日期之间相隔天数的平均值
    air_store_diff_time_mean = air_reserve_temp.groupby(['store_id', 'visit_date'])['diff_time'].agg(
        air_store_diff_time_mean='mean'
    )

    # air预定表，每天预定的入店日期与预定日期之间相隔天数的平均值
    air_diff_time_mean = air_reserve_temp.groupby(['visit_date'])['diff_time'].agg(
        air_diff_time_mean='mean'
    )

    air_result = air_result.unstack().fillna(0).stack()

    # air预定表，每天预定的总客户数和预定次数
    air_date_result = air_reserve_temp.groupby(['visit_date'])['reserve_visitors'].agg(
        air_date_visitors='sum',
        air_date_count='count'
    )

    # hpg预定表的时间窗口
    hpg_reserve_temp = hpg_reserve[(hpg_reserve.visit_date >= key[0])
                                   & (hpg_reserve.visit_date < label_end_date)
                                   & (hpg_reserve.reserve_date < key[0])].copy()

    # 预定的入店日期与预定日期之间相差的天数
    hpg_reserve_temp['diff_time'] = (pd.to_datetime(hpg_reserve['visit_datetime'])
                                     - pd.to_datetime(hpg_reserve['reserve_datetime'])).dt.days
    hpg_result = hpg_reserve_temp.groupby(['store_id', 'visit_date'])['reserve_visitors'].agg(
        hpg_reserve_visitors='sum',
        hpg_reserve_count='count'
    )

    hpg_result = hpg_result.unstack().fillna(0).stack()

    hpg_date_result = hpg_reserve_temp.groupby(['visit_date'])['reserve_visitors'].agg(
        hpg_date_visitors='sum',
        hpg_date_count='count'
    )

    hpg_store_diff_time_mean = hpg_reserve_temp.groupby(['store_id', 'visit_date'])['diff_time'].agg(
        hpg_store_diff_time_mean='mean'
    )

    hpg_diff_time_mean = hpg_reserve_temp.groupby(['visit_date'])['diff_time'].agg(
        hpg_diff_time_mean='mean'
    )

    air_result = left_merge(label, air_result, on=['store_id', 'visit_date']).fillna(0)
    air_store_diff_time_mean = left_merge(label, air_store_diff_time_mean, on=['store_id', 'visit_date']).fillna(0)
    hpg_result = left_merge(label, hpg_result, on=['store_id', 'visit_date']).fillna(0)
    hpg_store_diff_time_mean = left_merge(label, hpg_store_diff_time_mean, on=['store_id', 'visit_date']).fillna(0)
    air_date_result = left_merge(label, air_date_result, on=['visit_date']).fillna(0)
    air_diff_time_mean = left_merge(label, air_diff_time_mean, on=['visit_date']).fillna(0)
    hpg_date_result = left_merge(label, hpg_date_result, on=['visit_date']).fillna(0)
    hpg_diff_time_mean = left_merge(label, hpg_diff_time_mean, on=['visit_date']).fillna(0)
    result = pd.concat([air_result, hpg_result, air_date_result, hpg_date_result, air_store_diff_time_mean,
                        hpg_store_diff_time_mean, air_diff_time_mean, hpg_diff_time_mean], axis=1)
    return result


# second feature
def second_feat(result):
    result['store_mean_14_28_rate'] = result['store_mean14'] / (result['store_mean28'] + 0.01)
    result['store_mean_28_56_rate'] = result['store_mean28'] / (result['store_mean56'] + 0.01)
    result['store_mean_56_1000_rate'] = result['store_mean56'] / (result['store_mean1000'] + 0.01)
    result['genre_mean_28_56_rate'] = result['genre_mean28'] / (result['genre_mean56'] + 0.01)
    result['sgenre_mean_56_1000_rate'] = result['genre_mean56'] / (result['genre_mean1000'] + 0.01)
    return result


# 制作训练集
def make_feats(end_date, n_day):
    t0 = time.time()
    key = end_date, n_day
    print('data key为：{}'.format(key))
    print('add label')
    label = get_label(end_date, n_day)

    print('make feature...')
    result = [label]
    result.append(get_store_visitor_feat(label, key, 1000))  # store features
    result.append(get_store_visitor_feat(label, key, 56))  # store features
    result.append(get_store_visitor_feat(label, key, 28))  # store features
    result.append(get_store_visitor_feat(label, key, 14))  # store features
    result.append(get_store_exp_visitor_feat(label, key, 1000))  # store exp features
    result.append(get_store_week_feat(label, key, 1000))  # store dow features
    result.append(get_store_week_feat(label, key, 56))  # store dow features
    result.append(get_store_week_feat(label, key, 28))  # store dow features
    result.append(get_store_week_feat(label, key, 14))  # store dow features
    result.append(get_store_week_diff_feat(label, key, 58))  # store dow diff features
    result.append(get_store_week_diff_feat(label, key, 1000))  # store dow diff features
    result.append(get_store_all_week_feat(label, key, 1000))  # store all week feat
    result.append(get_store_week_exp_feat(label, key, 1000))  # store dow exp feat
    result.append(get_store_holiday_feat(label, key, 1000))  # store holiday feat

    result.append(get_genre_visitor_feat(label, key, 1000))  # genre feature
    result.append(get_genre_visitor_feat(label, key, 56))  # genre feature
    result.append(get_genre_visitor_feat(label, key, 28))  # genre feature
    result.append(get_genre_exp_visitor_feat(label, key, 1000))  # genre exp feature
    result.append(get_genre_week_feat(label, key, 1000))  # genre dow feature
    result.append(get_genre_week_feat(label, key, 56))  # genre dow feature
    result.append(get_genre_week_feat(label, key, 28))  # genre dow feature
    result.append(get_genre_week_exp_feat(label, key, 1000))  # genre dow exp feature

    result.append(get_reserve_feat(label, key))  # air_reserve, hpg_reserve
    result.append(get_first_last_time(label, key, 1000))  # first time and last time

    result.append(label)

    print('merge...')
    result = concat(result)

    result = second_feat(result)

    print('data shape：{}'.format(result.shape))
    print('spending {}s'.format(time.time() - t0))
    return result


train_feat = pd.DataFrame()
start_date = '2017-03-12'
for i in range(58):
    train_feat_sub = make_feats(date_add_days(start_date, i * (-7)), 39)
    train_feat = pd.concat([train_feat, train_feat_sub])
for i in range(1, 6):
    train_feat_sub = make_feats(date_add_days(start_date, i * 7), 42 - (i * 7))
    train_feat = pd.concat([train_feat, train_feat_sub])
test_feat = make_feats(date_add_days(start_date, 42), 39)

predictors = [f for f in test_feat.columns if
              f not in (['id', 'store_id', 'visit_date', 'end_date', 'air_area_name', 'visitors', 'month'])]

params = {
    'learning_rate': 0.02,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'sub_feature': 0.7,
    'num_leaves': 60,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}

t0 = time.time()
lgb_train = lgb.Dataset(train_feat[predictors], train_feat['visitors'])
lgb_test = lgb.Dataset(test_feat[predictors], test_feat['visitors'])

gbm = lgb.train(params, lgb_train, 2300)
pred = gbm.predict(test_feat[predictors])

print('训练用时{}秒'.format(time.time() - t0))
subm = pd.DataFrame({'id': test_feat.store_id + '_' + test_feat.visit_date, 'visitors': np.expm1(pred)})
subm = submission[['id']].merge(subm, on='id', how='left').fillna(0)
subm.to_csv(r'..\sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
            index=False, float_format='%.4f')
