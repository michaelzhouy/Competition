# -*- coding:utf-8 -*-
# Time   : 2020/4/29 22:56
# Email  : 15602409303@163.com
# Author : Zhou Yang

def correalation(df, threshold):
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
                colName = corr_matrix.columns[i]
                col_corr.add(colName)

    return col_corr

col = correalation(X_train, 0.8)
print('Correlated columns: ', col)
