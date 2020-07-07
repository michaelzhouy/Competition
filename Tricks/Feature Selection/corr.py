# -*- coding:utf-8 -*-
# Time   : 2020/4/29 22:56
# Email  : 15602409303@163.com
# Author : Zhou Yang


def correlation(df, threshold):
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
                colName_i = corr_matrix.columns[i]
                colName_j = corr_matrix.columns[j]
                if useful_cols[colName_i] >= useful_cols[colName_j]:
                    col_corr.add(colName_j)
                else:
                    col_corr.add(colName_i)

    return col_corr


col = correlation(df_train.drop(['phone_no_m', 'label'], axis=1), 0.98)
print('Correlated columns: ', col)
