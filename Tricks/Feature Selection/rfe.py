from sklearn.feature_selection import RFECV

lgb_model = lgb.LGBMRegressor(eval_metric='mae', random_state=666)

rfecv = RFECV(estimator=lgb_model,
              cv=5,
              step=1, # 每步删除的特征个数
              scoring='neg_mean_absolute_error', # 打分函数
              n_jobs=-1)
rfecv.fit(train_X, train_y)

print(rfecv.n_features_) # 选中的特征个数
print(rfecv.ranking_) # 特征排名

feats = list(np.array(train_X.columns)[rfecv.support_]) # 选中的特征
feats