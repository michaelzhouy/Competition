# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:36:36 2020

@author: z
"""

import sys
import numpy as np
import pandas as pd
import os 
import gc
from tqdm import tqdm, tqdm_notebook
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import LabelEncoder
import datetime
import time
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


path  = 'C:/Users/z/Python/Competition/DataFountain/乘用车细分市场销量预测/input/Round1/'
train_sales  = pd.read_csv(path+'train_sales_data.csv')
train_search = pd.read_csv(path+'train_search_data.csv')
train_user   = pd.read_csv(path+'train_user_reply_data.csv')
evaluation_public = pd.read_csv(path+'evaluation_public.csv')
submit_example    = pd.read_csv(path+'submit_example.csv')
data = pd.concat([train_sales, evaluation_public], ignore_index=True)
data = data.merge(train_search, 'left', on=['province', 'adcode', 'model', 'regYear', 'regMonth'])
data = data.merge(train_user, 'left', on=['model', 'regYear', 'regMonth'])
data['label'] = data['salesVolume']
# train_sales表中没有id
data['id'] = data['id'].fillna(0).astype(int)
# evaluation_public表中没有bodyType
data['bodyType'] = data['model'].map(train_sales.drop_duplicates('model').set_index('model')['bodyType'])
# LabelEncoder
for i in ['bodyType', 'model']:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(data[i].nunique()))))
data['mt'] = (data['regYear'] - 2016) * 12 + data['regMonth']


data['model_adcode'] = data['adcode'] + data['model']
data['model_adcode_mt'] = data['model_adcode'] * 100 + data['mt']

data['model_adcode_mt_label_1'] = data['model_adcode_mt'] + 1

data_last = data[data['label'].notnull()].set_index('model_adcode_mt_label_1')

data['shift_model_adcode_mt_label_1'] = data['model_adcode_mt'].map(data_last['label'])


def get_adjoin_feature(df_, start, end, col, group, space):
    '''
    相邻N月的首尾统计
    space: 间隔
    Notes: shift统一为adcode_model_mt
    '''
    df = df_.copy()
    add_feat = []
    for i in range(start, end+1):   
        add_feat.append('adjoin_{}_{}_{}_{}_{}_sum'.format(col,group,i,i+space,space)) # 求和
        add_feat.append('adjoin_{}_{}_{}_{}_{}_mean'.format(col,group,i,i+space,space)) # 均值
        add_feat.append('adjoin_{}_{}_{}_{}_{}_diff'.format(col,group,i,i+space,space)) # 首尾差值
        add_feat.append('adjoin_{}_{}_{}_{}_{}_ratio'.format(col,group,i,i+space,space)) # 首尾比例
        df['adjoin_{}_{}_{}_{}_{}_sum'.format(col,group,i,i+space,space)] = 0
        for j in range(0, space+1):
            df['adjoin_{}_{}_{}_{}_{}_sum'.format(col,group,i,i+space,space)]   = df['adjoin_{}_{}_{}_{}_{}_sum'.format(col,group,i,i+space,space)] +\
                                                                                  df['shift_{}_{}_{}'.format(col,'adcode_model_mt',i+j)]
        df['adjoin_{}_{}_{}_{}_{}_mean'.format(col,group,i,i+space,space)]  = df['adjoin_{}_{}_{}_{}_{}_sum'.format(col,group,i,i+space,space)].values/(space+1)
        df['adjoin_{}_{}_{}_{}_{}_diff'.format(col,group,i,i+space,space)]  = df['shift_{}_{}_{}'.format(col,'adcode_model_mt',i)].values -\
                                                                              df['shift_{}_{}_{}'.format(col,'adcode_model_mt',i+space)]
        df['adjoin_{}_{}_{}_{}_{}_ratio'.format(col,group,i,i+space,space)] = df['shift_{}_{}_{}'.format(col,'adcode_model_mt',i)].values /\
                                                                              df['shift_{}_{}_{}'.format(col,'adcode_model_mt',i+space)]
    return df, add_feat





