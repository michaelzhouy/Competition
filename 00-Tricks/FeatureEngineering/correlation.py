# -*- coding: utf-8 -*-
# @Time     : 2020/10/15 14:44
# @Author   : Michael_Zhouy


def correlation(df, threshold=0.98):
    """
    特征相关性计算
    @param df:
    @param threshold:
    @return:
    """
    col_corr = set()
    corr_matrix = df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colName = corr_matrix.columns[i]
                col_corr.add(colName)
    return col_corr


# 调用
col_corr = correlation(train, 0.98)
print(col_corr)
train.drop(list(col_corr), axis=1, inplace=True)
test.drop(list(col_corr), axis=1, inplace=True)