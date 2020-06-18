# -*- coding: utf-8 -*-
# @Time     : 2020/5/7 16:44
# @Author   : Michael_Zhouy

from sklearn.feature_selection import RFE, RFECV
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor  # 使用ExtraTrees 模型作为示范


# RFE
selector = RFE(estimator=clf, n_features_to_select=4, step=1)
# 与RFECV不同，此处RFE函数需要用户定义选择的变量数量，此处设置为选择4个最好的变量，每一步我们仅删除一个变量

selector = selector.fit(train_set, train_y) # 在训练集上训练

transformed_train = train_set[:,selector.support_]  # 转换训练集
assert np.array_equal(transformed_train, train_set[:,[0,5,6,7]]) # 选择了第一个，第六个，第七个及第八个变量

transformed_test = test_set[:,selector.support_] # 转换训练集
assert np.array_equal(transformed_test, test_set[:,[0,5,6,7]]) # 选择了第一个，第六个，第七个及第八个变量


# RFECV
clf = ExtraTreesRegressor(n_estimators=25)
selector = RFECV(estimator = clf, step = 1, cv = 5) # 使用5折交叉验证
# 每一步我们仅删除一个变量
selector = selector.fit(train_set, train_y)

transformed_train = train_set[:,selector.support_]  # 转换训练集
assert np.array_equal(transformed_train, train_set) # 选择了所有的变量

transformed_test = test_set[:,selector.support_] # 转换训练集
assert np.array_equal(transformed_test, test_set) # 选择了所有的变量


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
backward_model.k_feature_idx_
X_train.columns[list(backward_model.k_feature_idx_)]


#
emodel = ExhaustiveFeatureSelector(RandomForestRegressor(),
                                   min_features=1,
                                   max_features=5,
                                   scoring='r2',
                                   n_jobs=-1)
miniData=X_train[X_train.columns[list(backward_model.k_feature_idx_)]]
emodel.fit(miniData, y_train)
emodel.best_idx_
miniData.columns[list(emodel.best_idx_)]


