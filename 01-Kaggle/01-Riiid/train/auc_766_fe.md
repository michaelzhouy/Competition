## get_state
1. 

## get_user_data
1. 

```python
user_tmp = train.groupby('user_id', as_index=False)['answered_correctly'].agg({
    'user_answered_correctly_mean': 'mean',
    'user_answered_correctly_sum': 'sum',
    'user_answered_correctly_cnt': 'count'})
train.merge(tmp, on='user_id', on='left')

tmp = 
```