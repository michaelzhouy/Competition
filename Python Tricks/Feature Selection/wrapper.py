# -*- coding: utf-8 -*-
# @Time     : 2020/5/7 16:44
# @Author   : Michael_Zhouy

from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.ensemble import RandomForestRegressor

# 前向特征选择
forward_model = SequentialFeatureSelector(RandomForestRegressor(),
                                  k_features=10,
                                  forward=True,
                                  verbose=2,
                                  cv=5,
                                  n_jobs=-1,
                                  scoring='r2')
model.fit(X_train, y_train)
model.k_feature_idx_
model.k_feature_names_


# 后向特征选择
backward_model = SequentialFeatureSelector(RandomForestRegressor(),
                                           k_features=10,
                                           forward=False,
                                           verbose=2,
                                           cv=5,
                                           n_jobs=-1,
                                           scoring='r2')
backward_model.fit(X_train, y_train)
backwardModel.k_feature_idx_
X_train.columns[list(backwardModel.k_feature_idx_)]


#
emodel = ExhaustiveFeatureSelector(RandomForestRegressor(),
                                   min_features=1,
                                   max_features=5,
                                   scoring='r2',
                                   n_jobs=-1)
miniData=X_train[X_train.columns[list(backwardModel.k_feature_idx_)]]
emodel.fit(miniData, y_train)
emodel.best_idx_
miniData.columns[list(emodel.best_idx_)]