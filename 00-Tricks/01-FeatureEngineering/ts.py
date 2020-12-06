# -*- coding: utf-8 -*-
# @Time     : 2020/11/10 21:14
# @Author   : Michael_Zhouy
import numpy as np
import pandas as pd
import gc


def group_rolling(df, num_cols):
    """
    分组rolling
    @param df:
    @param num_cols:
    @return:
    """
    for i in num_cols:
        for j in ['90min', '120min']:
            df.set_index('datetime', inplace=True)
            tmp = df[i].rolling(j, closed='left', min_periods=1).agg({
                '{}_{}_rolling_mean'.format(i, j): 'mean',
                '{}_{}_rolling_median'.format(i, j): 'median',
                '{}_{}_rolling_max'.format(i, j): 'max',
                '{}_{}_rolling_min'.format(i, j): 'min',
                '{}_{}_rolling_sum'.format(i, j): 'sum',
                '{}_{}_rolling_std'.format(i, j): 'std',
                '{}_{}_rolling_skew'.format(i, j): 'skew'
            })
            tmp.reset_index(inplace=True)
            df.reset_index(inplace=True)
            df = df.merge(tmp, on=['datetime'], how='left')
            del tmp
            gc.collect()
    return df
