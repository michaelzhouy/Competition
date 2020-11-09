# -*- coding: utf-8 -*-
# @Time     : 2020/11/9 15:45
# @Author   : Michael_Zhouy
import numpy as np
from sklearn.impute import SimpleImputer


# 1. 用众数填充缺失值
df['col'].fillna(df['col'].mode()[0], inplace=True)

# 2. 用分组后的众数填充缺失值
df['col1'] = df.groupby(['col2', 'col3'])['col1'].transform(lambda x: x.fillna(x.mode()[0]))`

# 3. 用分组后的中位数填充缺失值
df['col1'] = df.groupby(['col2', 'col3'])['col1'].transform(lambda x: x.fillna(x.median()))

# 4. 用每列的众数填充每列的缺失值
df = df.fillna(df.mode().iloc[0, :])


def fillna_median(train, test):
    """
    用每列中位数填充缺失值
    @param train:
    @param test:
    @return:
    """
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    imputer = imputer.fit(train)
    train_imputer = imputer.transform(train)
    test_imputer = imputer.transform(test)
    return train_imputer, test_imputer
