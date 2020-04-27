from sklearn.feature_selection import RFECV

lgb_model = lgb.LGBMRegressor(eval_metric='mae', random_state=666)

rfecv = RFECV(estimator=lgb_model, cv=5, scoring='neg_mean_absolute_error')
rfecv.fit(train_X, train_y)
print(rfecv.n_features_)
print(rfecv.ranking_)
print(rfecv.grid_scores_)

feats = list(np.array(train_X.columns)[rfecv.support_])
feats