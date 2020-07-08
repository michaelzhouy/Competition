# -*- coding:utf-8 -*-
# Time   : 2020/5/10 19:07
# Email  : 15602409303@163.com
# Author : Zhou Yang

import numpy as np
import pandas as pd
import os
import gc

# 过滤警告
import warnings
warnings.filterwarnings('ignore')

# DataFrame显示所有列
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)

# 读取压缩文件
df = pd.read_csv('../input/file.txt.gz', compression='gzip', header=0, sep=',', quotechar='"')

# 数据存储为h5格式
df.to_hdf('data.h5', 'df')
# 读取h5文件
pd.read_hdf('data.h5')


# 节省内存读文件
def reduce_mem_usage(df):
    """
    iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    @param df:
    @return:
    """
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
            df[col] = df[col].astype('str')

    end_mem = df.memory_usage().sum()
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


# 读取文件下所有的文件，并合并成一个文件
def read_data(path):
    data_list = []
    for f in os.listdir(path):
        print(f)
        df = pd.read_csv(path + os.sep + f)
        print(df.shape)
        data_list.append(df)
        del df
        gc.collect()

    res = pd.concat(data_list, ignore_index=True)
    return res
