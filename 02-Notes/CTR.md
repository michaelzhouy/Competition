# CTR

## EDA

1.  查看各个特征的含义

2.  uid组成

3.  数据是否平衡，train['label'].mean()

4.  类别特征、数值特征、时间特征，train.nunique()，取值较多的更可能为数值特征

5.  类别特征中，在test中出现，在train中没有出现的情况

## Baseline

   1. 对类别特征做LabelEncoder编码

   2. 对类别特征做Count编码

   3. 对在test中出现，在train中没有出现的取值进行编码（冷启动问题）
```python
def get_same_set(train_df,test_df):
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
```

   4. 类别层次特征，对类似的类别特征进行拼接，最后做LabelEncoder
   5. 对时间特征取几时，几日，周几等
   6. 5折交叉验证
```python
pred_lgb = []
for i in range(5):
    print(i)
    train_X, val_X,train_Y, val_Y = train_test_split(train_data[train_cols].values, train_data[y].values, test_size=0.15, random_state=i)
    dtrain = lgb.Dataset(train_X,train_Y)
    dval = lgb.Dataset(val_X,val_Y,reference=dtrain) 
    lgb_model_1 = lgb.train(params, dtrain,
                            num_boost_round=160,
                            valid_sets=[dtrain, dval],
                            verbose_eval=20)   
    pred_lgb.append(lgb_model_1.predict(test_data[train_cols].values)) 

submit_sample = pd.read_csv('./Data/sampleSubmission.csv')
submit_sample['click'] =  np.mean(pred_lgb, axis=0)
submit_sample[['id','click']].to_csv('./sub/baseline5.csv', index=False)
```

