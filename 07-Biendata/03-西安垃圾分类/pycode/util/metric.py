# -*- coding: utf-8 -*-
# @Time    : 2021/9/23 9:37 上午
# @Author  : Michael Zhouy
from sklearn.metrics import f1_score


def eval_metric(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')
