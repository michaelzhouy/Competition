# -*- coding: utf-8 -*-
# @Time     : 2020/10/19 14:19
# @Author   : Michael_Zhouy
import numpy as np
from sklearn.metrics import mean_squared_log_error


def rmsle(y_hat, data):
    y_true = data.get_label()
    y_hat = np.where(y_hat < 0, 1, y_hat)
    y_true = np.where(y_true < 0, 1, y_hat)
    res = np.sqrt(mean_squared_log_error(y_true, y_hat))
    return 'rmsle', res, True


# regression
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.05,
    'verbose': -1,
    'seed': 2020
}
