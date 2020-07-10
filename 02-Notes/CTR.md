# CTR
## Part1
### EDA

1.  查看各个特征的含义

2.  uid组成

3.  数据是否平衡，train['label'].mean()

4.  类别特征、数值特征、时间特征，train.nunique()，取值较多的更可能为数值特征

5.  类别特征中，在test中出现，在train中没有出现的情况

### Baseline

      1. 对类别特征做LabelEncoder编码
      2. 对类别特征做Count编码
      3. 对在test中出现，在train中没有出现的取值进行编码（冷启动问题）
```python
def get_same_set(train_df,test_df):
    """
    将train和test的取值的交集编码为较小的值，它们之间的取值的差集编码为较大的值
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
```

   4. 类别层次特征，对类似的类别特征进行拼接，最后做LabelEncoder
   5. 对时间特征取几时，几日，周几等
   6. 5折交叉验证
```python
pred_lgb = []
for i in range(5):
    print(i)
    train_X, val_X,train_Y, val_Y = train_test_split(train_data[train_cols], train_data[y], test_size=0.15, random_state=i)
    dtrain = lgb.Dataset(train_X, train_Y)
    dval = lgb.Dataset(val_X, val_Y, reference=dtrain) 
    lgb_model = lgb.train(params,
                            dtrain,
                            num_boost_round=160,
                            valid_sets=[dtrain, dval],
                            verbose_eval=20)   
    pred_lgb.append(lgb_model.predict(test_data[train_cols].val)) 

submit_sample = pd.read_csv('../input/sampleSubmission.csv')
submit_sample['click'] =  np.mean(pred_lgb, axis=0)
submit_sample[['id','click']].to_csv('./sub/baseline5.csv', index=False)
```

---

## Part 2

1.  对类别特征value_counts()，查看异常情况，是否某些取值数量高出其他取值数量几个数量级，可能是缺失值的一种编码方式

2.  基于用户的统计特征（找到uid）

     1.    时间序列，用户每天，每个小时在app上出现的次数，用户距上一次出现的时间差

           ```python
           data['datetime'] = data['hour'].map(lambda x: datetime.strptime(str(x), '%y%m%d%H'))
           
           data['dayoftheweek'] = data['datetime'].map(lambda x: x.weekday())
           data['day'] = data['datetime'].map(lambda x: x.day)
           data['hour'] = data['datetime'].map(lambda x: x.hour)
           
           data['time'] = (data['day'] - data['day'].min()) * 24  + data['hour']
           
           groupby(['uid', 'day'])['label'].agg(xxx='count')
           groupby(['uid', 'time'])['label'].agg(xxx='count')
           ```

     2.    时间序列，用户每天、每个小时

           ```python
           groupby(['uid', 'day', 'C14'])['label'].agg(xxx='count')
           groupby(['uid', 'time', 'C17'])['label'].agg(xxx='count')
           ```

           
