# -*- coding: utf-8 -*-
# @Time     : 2020/5/7 20:06
# @Author   : Michael_Zhouy

import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import itertools
import warnings
from sklearn.model_selection import StratifiedKFold, KFold
from itertools import product
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
from sklearn.feature_selection import RFECV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum()
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum()
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


class MeanEncoder:
    def __init__(self, categorical_features, n_splits=10, target_type='classification', prior_weight_func=None):
        """
        :param categorical_features: list of str, the name of the categorical columns to encode

        :param n_splits: the number of splits used in mean encoding

        :param target_type: str, 'regression' or 'classification'

        :param prior_weight_func:
        a function that takes in the number of observations, and outputs prior weight
        when a dict is passed, the default exponential decay function will be used:
        k: the number of observations needed for the posterior to be weighted equally as the prior
        f: larger f --> smaller slope
        """

        self.categorical_features = categorical_features
        self.n_splits = n_splits
        self.learned_stats = {}

        if target_type == 'classification':
            self.target_type = target_type
            self.target_values = []
        else:
            self.target_type = 'regression'
            self.target_values = None

        if isinstance(prior_weight_func, dict):
            self.prior_weight_func = eval('lambda x: 1 / (1 + np.exp((x - k) / f))', dict(prior_weight_func, np=np))
        elif callable(prior_weight_func):
            self.prior_weight_func = prior_weight_func
        else:
            self.prior_weight_func = lambda x: 1 / (1 + np.exp((x - 2) / 1))

    @staticmethod
    def mean_encode_subroutine(X_train, y_train, X_test, variable, target, prior_weight_func):
        X_train = X_train[[variable]].copy()
        X_test = X_test[[variable]].copy()

        if target is not None:
            nf_name = '{}_pred_{}'.format(variable, target)
            X_train['pred_temp'] = (y_train == target).astype(int)  # classification
        else:
            nf_name = '{}_pred'.format(variable)
            X_train['pred_temp'] = y_train  # regression
        prior = X_train['pred_temp'].mean()

        col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg(['mean', 'size'])
        col_avg_y['size'] = prior_weight_func(col_avg_y['size'])
        col_avg_y[nf_name] = col_avg_y['size'] * prior + (1 - col_avg_y['size']) * col_avg_y['mean']
        col_avg_y.drop(['size', 'mean'], axis=1, inplace=True)

        nf_train = X_train.join(col_avg_y, on=variable)[nf_name].values
        nf_test = X_test.join(col_avg_y, on=variable).fillna(prior, inplace=False)[nf_name].values

        return nf_train, nf_test, prior, col_avg_y

    def fit_transform(self, X, y):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :param y: pandas Series or numpy array, n_samples
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()
        if self.target_type == 'classification':
            skf = StratifiedKFold(self.n_splits)
        else:
            skf = KFold(self.n_splits)

        if self.target_type == 'classification':
            self.target_values = sorted(set(y))
            self.learned_stats = {'{}_pred_{}'.format(variable, target): [] for variable, target in
                                  product(self.categorical_features, self.target_values)}
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(X, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, target,
                        self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        else:
            self.learned_stats = {'{}_pred'.format(variable): [] for variable in self.categorical_features}
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(X, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, None,
                        self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        return X_new

    def transform(self, X):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()

        if self.target_type == 'classification':
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits
        else:
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits

        return X_new


def date_proc(x):
    # '20200426' '20200026'
    m = int(x[4:6])
    if m == 0:
        m = 1
    return x[:4] + '-' + str(m) + '-' + x[6:]


# 定义日期提取函数
def date_tran(df,fea_col):
    for f in tqdm(fea_col):
        df[f] = pd.to_datetime(df[f].astype('str').apply(date_proc))
        df[f + '_year'] = df[f].dt.year # 年份
        df[f + '_month'] = df[f].dt.month # 月份
        df[f + '_day'] = df[f].dt.day # 多少号
        df[f + '_dayofweek'] = df[f].dt.dayofweek # 周几
    return df


# 分桶操作
def cut_group(df, cols, num_bins=50):
    for col in cols:
        all_range = int(df[col].max() - df[col].min())
        bin = [i * all_range / num_bins for i in range(all_range)]
        df[col + '_bin'] = pd.cut(df[col], bin, labels=False)
    return df


# count编码
def count_coding(df, fea_col):
    for f in fea_col:
        df[f + '_count'] = df[f].map(df[f].value_counts())
    return df


# 定义交叉特征统计
def cross_cat_num(df, num_col, cat_col):
    for f1 in tqdm(cat_col):
        g = df.groupby(f1, as_index=False)
        for f2 in tqdm(num_col):
            feat = g[f2].agg({
                '{}_{}_max'.format(f1, f2): 'max', # 最大值
                '{}_{}_min'.format(f1, f2): 'min', # 最小值
                '{}_{}_median'.format(f1, f2): 'median', # 中位数
            })
            df = df.merge(feat, on=f1, how='left')
    return df


Train_data = reduce_mem_usage(pd.read_csv('input/used_car_train_20200313.csv',
                                          sep=' '))
TestA_data = reduce_mem_usage(pd.read_csv('input/used_car_testB_20200421.csv',
                                          sep=' '))

# 合并数据集
concat_data = pd.concat([Train_data, TestA_data],
                        ignore_index=True)  # 重新生成索引

# 'notRepairedDamage'中的'-'用0替换
concat_data['notRepairedDamage'] = concat_data['notRepairedDamage'].replace('-', 0).astype('float16')

# 处理异常值
concat_data['power'][concat_data['power'] > 600] = 600
concat_data['power'][concat_data['power'] < 1] = 1

concat_data['v_13'][concat_data['v_13'] > 6] = 6
concat_data['v_14'][concat_data['v_14'] > 4] = 4

# v系列特征之间相加
for j in ['v_' + str(i) for i in range(14)]:
    for k in ['v_' + str(m) for m in range(14)]:
        concat_data[j + '+' + k] = concat_data[j] + concat_data[k]

# 原始特征与v系列特征之间相乘
for i in ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'power', 'kilometer', 'notRepairedDamage', 'regionCode']:
    for j in ['v_' + str(k) for k in range(14)]:
        concat_data[i + '*' + j] = concat_data[i] * concat_data[j]

# 提取日期信息
date_cols = ['regDate', 'creatDate']
concat_data = date_tran(concat_data, date_cols)

data = concat_data.copy()

# count编码
count_list = ['regDate', 'creatDate', 'model', 'brand', 'regionCode', 'bodyType', 'fuelType', 'name',
              'regDate_year', 'regDate_month', 'regDate_day', 'regDate_dayofweek',
              'creatDate_month', 'creatDate_day', 'creatDate_dayofweek', 'kilometer']

data = count_coding(data, count_list)

data['used_time1'] = (pd.to_datetime(data['creatDate'], format='%Y%m%d', errors='coerce') -
                      pd.to_datetime(data['regDate'], format='%Y%m%d', errors='coerce')).dt.days
data['used_time2'] = (pd.datetime.now() - pd.to_datetime(data['regDate'], format='%Y%m%d', errors='coerce')).dt.days
data['used_time3'] = (pd.datetime.now() - pd.to_datetime(data['creatDate'], format='%Y%m%d', errors='coerce') ).dt.days

# 用数值特征对类别特征做统计刻画，随便挑了几个跟price相关性最高的匿名特征
cross_cat = ['model', 'brand', 'regDate_year']
cross_num = ['v_0', 'v_3', 'v_4', 'v_8', 'v_12', 'power', 'used_time1']
data = cross_cat_num(data, cross_num, cross_cat)  # 一阶交叉
# 选择特征列
numerical_cols = data.columns

cat_fea = ['SaleID', 'offerType', 'seller']
feature_cols = [col for col in numerical_cols if col not in cat_fea]
feature_cols = [col for col in feature_cols if col not in ['price']]

# 将训练集和测试集分开
X_data = data.iloc[:len(Train_data), :][feature_cols]
Y_data = Train_data['price']
X_test = data.iloc[len(Train_data):, :][feature_cols]

class_list = ['model', 'brand', 'name', 'regionCode'] + date_cols
MeanEncodeFeature = class_list
ME = MeanEncoder(categorical_features=MeanEncodeFeature, n_splits=5, target_type='regression', prior_weight_func=None)
X_data = ME.fit_transform(X_data, Y_data)
X_test = ME.transform(X_test)

X_data['price'] = Train_data['price']

# target encoding目标编码，回归场景相对来说做目标编码的选择更多，不仅可以做均值编码，还可以做标准差编码、中位数编码等
enc_cols = []
stats_default_dict = {
    'max': X_data['price'].max(),
    'min': X_data['price'].min(),
    'median': X_data['price'].median(),
    'mean': X_data['price'].mean(),
    'sum': X_data['price'].sum(),
    'std': X_data['price'].std(),
    'skew': X_data['price'].skew(), # 偏度
    'kurt': X_data['price'].kurt(), # 峰度
    'mad': X_data['price'].mad() # mean absolute deviation 平均绝对偏差
}

# 暂且选择这三种编码
enc_stats = ['max', 'min', 'mean']
skf = KFold(n_splits=10, shuffle=True, random_state=42)
for f in tqdm(['regionCode', 'brand', 'regDate_year' ,'creatDate_year', 'kilometer', 'model']):
    enc_dict = {}
    for stat in enc_stats:
        enc_dict['{}_target_{}'.format(f, stat)] = stat
        X_data['{}_target_{}'.format(f, stat)] = 0
        X_test['{}_target_{}'.format(f, stat)] = 0
        enc_cols.append('{}_target_{}'.format(f, stat))
    for i, (trn_idx, val_idx) in enumerate(skf.split(X_data, Y_data)):
        trn_x, val_x = X_data.iloc[trn_idx].reset_index(drop=True), X_data.iloc[val_idx].reset_index(drop=True)
        enc_df = trn_x.groupby(f, as_index=False)['price'].agg(enc_dict)
        val_x = val_x[[f]].merge(enc_df, on=f, how='left')
        test_x = X_test[[f]].merge(enc_df, on=f, how='left')
        for stat in enc_stats:
            val_x['{}_target_{}'.format(f, stat)] = val_x['{}_target_{}'.format(f, stat)].fillna(stats_default_dict[stat])
            test_x['{}_target_{}'.format(f, stat)] = test_x['{}_target_{}'.format(f, stat)].fillna(stats_default_dict[stat])
            X_data.loc[val_idx, '{}_target_{}'.format(f, stat)] = val_x['{}_target_{}'.format(f, stat)].values
            X_test['{}_target_{}'.format(f, stat)] += test_x['{}_target_{}'.format(f, stat)].values / skf.n_splits

drop_list = ['regDate', 'creatDate', 'brand_power_min', 'regDate_year_power_min']
x_train = X_data.drop(drop_list + ['price'], axis=1)
x_test = X_test.drop(drop_list, axis=1)
print('x_train.shape: ', x_train.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

lgb_model = lgb.LGBMRegressor(eval_metric='mae', random_state=666)

# 后向特征选择
backward_model = SFS(lgb.LGBMRegressor(objective='regression_l1', random_state=666),
                     k_features=100,
                     forward=False,
                     verbose=2,
                     cv=5,
                     n_jobs=-1,
                     scoring='neg_mean_absolute_error')
backward_model.fit(x_train, Train_data['price'])
cols = x_train.columns[list(backward_model.k_feature_idx_)]
print('cols: ', cols)

params = {'objective': 'regression_l1',
          'boosting': 'gbdt',
          'metric': 'mae',
          'learning_rate': 0.1,
          'num_leaves': 31,
          'seed': 2020}

lgb_train = lgb.Dataset(x_train[cols], label=Train_data['price'])

cv_results = lgb.cv(params,
                    lgb_train,
                    num_boost_round=100000,
                    early_stopping_rounds=200,
                    eval_train_metric=True)
cv_df = pd.DataFrame(cv_results)
print(cv_df)
