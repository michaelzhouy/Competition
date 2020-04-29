# -*- coding:utf-8 -*-
# Time   : 2020/4/11 20:10
# Email  : 15602409303@163.com
# Author : Zhou Yang

from category_encoders import cat_boost
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
X = pd.DataFrame({'A': [1, 1, 1, 2, 2, 3, 4, 5], 'B': ['a', 'b', 'a', 'c', 'd', 'a', 'b', 'e']})
y = pd.Series([2, 3, 8, 9, 5, 8, 4, 2])

y_oof = np.zeros(X.shape[0])
y_test_oof = np.zeros(test.shape[0]).reshape(-1,1)
folds = KFold(2)

for train_index, valid_index in folds.split(X, y):
    X_train, X_valid = X['B'].iloc[train_index], X['B'].iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    clf = cat_boost.CatBoostEncoder()
    clf.fit(X_train, y_train)
    y_pred_valid = clf.transform(X_valid)
    y