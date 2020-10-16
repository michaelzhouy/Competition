# -*- coding:utf-8 -*-
# Time   : 2020/5/10 18:50
# Email  : 15602409303@163.com
# Author : Zhou Yang

import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import os
from scipy import stats
from joblib import Parallel, delayed
from datetime import datetime
from mlxtend.feature_selection import SequentialFeatureSelector

# 过滤警告
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

# DataFrame显示所有列
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)

# 筛选object特征
df_object = df.select_dtypes(include=['object'])
df_numerical = df.select_dtypes(exclude=['object'])


# 分组排名
df.groupby('uid')['time'].rank('dense')


# groupby
gb = df.groupby(['user_id', 'page_id'], ax_index=False).agg(
    {'ad_price': {'max_price': np.max, 'min_price': np.min}})
gb.columns = ['user_id', 'page_id', 'min_price', 'max_price']
df = pd.merge(df, gb, on=['user_id', 'page_id'], how='left')


# rolling时间窗口
df['TTI'].groupby(df['id_road']).rolling('60min', closed='left', min_periods=6).mean()


"""
通话时间点的偏好
"""

# hour通话次数最高
tmp = df_voc.groupby('phone_no_m')['voc_hour'].agg(voc_hour_mode=lambda x: stats.mode(x)[0][0],        # 频次最高的元素
                                                   voc_hour_mode_count=lambda x: stats.mode(x)[1][0],  # 频次最高的元素的频次
                                                   voc_hour_nunique='nunique')
phone_no_m = phone_no_m.merge(tmp, on='phone_no_m', how='left')

# day通话次数最高
tmp = df_voc.groupby('phone_no_m')['voc_day'].agg(voc_day_mode=lambda x: stats.mode(x)[0][0],
                                                  voc_day_mode_count=lambda x: stats.mode(x)[1][0],
                                                  voc_day_nunique='nunique')
phone_no_m = phone_no_m.merge(tmp, on='phone_no_m', how='left')


# 每天的通话次数
voc_day_cnt_res = df_voc.groupby(['phone_no_m', 'voc_day'])['phone_no_m'].count().unstack()
for i in df_voc['voc_day'].unique():
    phone_no_m['voc_day{}_count'.format(i)] = phone_no_m['phone_no_m'].map(voc_day_cnt_res[i])

# 每天的通话人数
voc_day_nunique_res = df_voc.groupby(['phone_no_m', 'voc_day'])['opposite_no_m'].nunique().unstack()
for i in df_voc['voc_day'].unique():
    phone_no_m['voc_day{}_nunique'.format(i)] = phone_no_m['phone_no_m'].map(voc_day_nunique_res[i])

# 每天的通话时长
voc_day_call_dur_res = df_voc.groupby(['phone_no_m', 'voc_day'])['call_dur'].sum().unstack()
for i in df_voc['voc_day'].unique():
    phone_no_m['voc_day{}_call_dur_sum'.format(i)] = phone_no_m['phone_no_m'].map(voc_day_call_dur_res[i])