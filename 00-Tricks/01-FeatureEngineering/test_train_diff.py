# -*- coding: utf-8 -*-
# @Time     : 2020/10/15 14:54
# @Author   : Michael_Zhouy


def get_same_set(train_df, test_df):
    """
    test中出现，train中没有出现的取值编码
    @param train_df:
    @param test_df:
    @return:
    """
    train_diff_test = set(train_df) - set(test_df)
    same = set(train_df) - train_diff_test
    test_diff_train = set(test_df) - same
    dic_ = {}
    cnt = 0
    for val in same:
        dic_[val] = cnt
        cnt += 1
    for val in train_diff_test:
        dic_[val] = cnt
        cnt += 1
    for val in test_diff_train:
        dic_[val] = cnt
        cnt += 1
    return dic_


for col in data.columns:
    if col != 'id' and col != 'click':
        if data[col].dtypes == 'O':
            print(col)
            dic_ = get_same_set(train[col].values, test[col].values)
            data[col+'_zj_encode'] = data[col].apply(lambda x: dic_[x])
