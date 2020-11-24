!pip install datatable==0.11.0 > /dev/null

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from multiprocessing import cpu_count
from tqdm.notebook import tqdm
import datatable as dt
from tqdm import tqdm
import gc
import lightgbm as lgb

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
            df[col] = df[col].astype('str')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


data_types_dict = {
    'user_id': 'int32',
    'content_id': 'int16',
    'answered_correctly': 'int8',
    'prior_question_elapsed_time': 'float32',
    'prior_question_had_explanation': 'bool'
}
train_df = reduce_mem_usage(dt.fread('../input/riiid-test-answer-prediction/train.csv', columns=set(data_types_dict.keys())).to_pandas())
display(train_df.head())


# 课程数据删除
train_df = train_df[train_df['answered_correctly'] != -1].reset_index(drop=True)
train_df['prior_question_had_explanation'].fillna(False, inplace=True)
train_df = train_df.astype(data_types_dict)
gc.collect()


train_df['user_content_id'] = train_df['user_id'] * 100000 + train_df['content_id']


questions_df = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv')
questions_df = questions_df[['question_id', 'part']]
questions_df.rename(columns={'question_id': 'content_id'}, inplace=True)
train_df = train_df.merge(questions_df, on='content_id', how='left')
del questions_df
gc.collect()


def get_index_np(N_ROWS = 99271300):
    return np.arange(N_ROWS) # speeds up dataframe creation

res = {}
content_mean = train_df.groupby('content_id')['answered_correctly'].agg(['mean'])['mean'].to_dict()
train_df['mean_content_accuracy'] = train_df['content_id'].map(content_mean)
res['content_mean'] = content_mean
del content_mean
gc.collect()

user = train_df.groupby('user_id')['answered_correctly'].agg(['mean', 'sum', 'count'])
user_mean = user['mean'].to_dict()
user_sum = user['sum'].to_dict()
user_count = user['count'].to_dict()
train_df['mean_user_accuracy'] = train_df['user_id'].map(user_mean)
train_df['answered_correctly_user'] = train_df['user_id'].map(user_sum)
train_df['answered_user'] = train_df['user_id'].map(user_count)
res['user_mean'] = user_mean
res['user_sum'] = user_sum
res['user_count'] = user_count
del user, user_mean, user_sum, user_count
gc.collect()

user_content_count = train_df.groupby('user_content_id')['content_id'].agg(['count'])['count'].to_dict()
train_df['attempt'] = train_df['user_content_id'].map(user_content_count)
res['user_content_count'] = user_content_count
del user_content_count
gc.collect()

train_df['hmean_user_content_accuracy'] = 2 * (
        (train_df['mean_user_accuracy'] * train_df['mean_content_accuracy']) /
        (train_df['mean_user_accuracy'] + train_df['mean_content_accuracy'])
)

cols = ['prior_question_elapsed_time', 'mean_user_accuracy', 'answered_correctly_user', 'answered_user'
        'mean_content_accuracy', 'part', 'hmean_user_content_accuracy', 'attempt',
        'prior_question_had_explanation', 'answered_correctly']
train_df = train_df[cols]
gc.collect()


train_df = train_df.groupby('user_id').tail(24).reset_index(drop=True)
gc.collect()

for i in ['part', 'prior_question_had_explanation']:
    train_df[i] = train_df[i].astype('category')

valid_df = train_df.groupby('user_id').tail(6).reset_index(drop=True)
train_df.drop(valid_df.index, inplace=True)
gc.collect()

used_cols = ['prior_question_elapsed_time', 'mean_user_accuracy', 'answered_correctly_user', 'answered_user'
             'mean_content_accuracy', 'part', 'hmean_user_content_accuracy', 'attempt',
             'prior_question_had_explanation']

y_train = train_df['answered_correctly']
X_train = train_df[used_cols]
y_valid = valid_df['answered_correctly']
X_valid = valid_df[used_cols]
gc.collect()

dtrain = lgb.Dataset(X_train, y_train)
dvalid = lgb.Dataset(X_valid, y_valid, reference=dtrain)

del X_train, X_valid, y_train, y_valid
gc.collect()

params = {
    'objective': 'binary',
    'metric': 'auc'
}


def train():
    evals_result = {}
    model = lgb.train(
        params=params,
        train_set=dtrain,
        valid_sets=[dvalid],
        num_boost_round=2500,
        verbose_eval=10,
        evals_result=evals_result,
        early_stopping_rounds=10
    )

    # save model
    model.save_model('lgb.txt')

    return model, evals_result


model, evals_result = train()


# def get_user_data(state, test_df):
#     # updated data
#     attempt, mean_user_accuracy, answered_correctly_user, answered_user = [], [], [], []
#     for idx, (user_id, content_id, user_content_id) in test_df[['user_id', 'content_id', 'user_content_id']].iterrows():
#         # check if user exists
#         if user_id in res['user_id']['mean'].index:
#             # check if user already answered the question, if so update it to a maximum of 4
#             if user_content_id in res['user_content_id']['count'].index:
                
#                 state[user_id]['user_content_attempts'][content_id] = min(4, state[user_id]['user_content_attempts'][
#                     content_id] + 1)
#             # if user did not answered the question already, set the number of attempts to 0
#             else:
#                 state[user_id]['user_content_attempts'][content_id] = 0

#         # else create user with default values
#         else:
#             dict_keys = ['mean_user_accuracy', 'answered_correctly_user', 'answered_user', 'user_content_attempts']
#             dict_default_vals = [0.680, 0, 0, dict(zip([content_id], [0]))]
#             state[user_id] = dict(zip(dict_keys, dict_default_vals))

#         # add user data to lists
#         attempt.append(state[user_id]['user_content_attempts'][content_id])
#         mean_user_accuracy.append(state[user_id]['mean_user_accuracy'])
#         answered_correctly_user.append(state[user_id]['answered_correctly_user'])
#         answered_user.append(state[user_id]['answered_user'])

#     return attempt, mean_user_accuracy, answered_correctly_user, answered_user

def get_mean_user_accuracy(user_id):
    if user_id in res['user_mean']:
        return res['user_mean'][user_id]
    else:
        return 0.68


def get_answered_correctly_user(user_id):
    if user_id in res['user_sum']:
        return res['user_sum'][user_id]
    else:
        return 0


def get_answered_user(user_id):
    if user_id in res['user_count']:
        return res['user_count'][user_id]
    else:
        return 0


def get_user_content_attempts(user_content_id):
    if user_content_id in res['user_content_id']:
        return min(5, res['user_content_id'][user_content_id])
    else:
        return 0


# updates the user data
def update_user_data(state, prev_test_df):
    for user_id, content_id, answered_correctly in prev_test_df[['user_id', 'content_id', 'answered_correctly']].values:
        res['user_sum'][user_id] += answered_correctly
        res['user_count'][user_id] += 1
        res['user_mean'][user_id] = res['user_count'][user_id] / res['user_sum'][user_id]

env = riiideducation.make_env()
iter_test = env.iter_test()

prev_test_df = None


for idx, (test_df, _) in tqdm(enumerate(iter_test)):
    
    # merge with all features
    test_df['user_content_id'] = test_df['user_id'] * 100000 + test_df['content_id']
    test_df = test_df.merge(questions_df, on='content_id', how='left')
    test_df['mean_content_accuracy'] = test_df['content_id'].map(res['content_id']['mean'])
    
    # from 2nd iteration, update user data
    if prev_test_df is not None:
        prev_test_df['answered_correctly'] = eval(test_df['prior_group_answers_correct'].iloc[0])
        update_user_data(state, prev_test_df.loc[prev_test_df['content_type_id'] == 0])
        if idx is 1:
            display(test_df)
            display(prev_test_df)

    test_df['mean_user_accuracy'] = test_df['user_id'].map(get_mean_user_accuracy)
    test_df['answered_correctly_user'] = test_df['user_id'].map(get_answered_correctly_user)
    test_df['answered_user'] = test_df['user_id'].map(get_answered_user)
    test_df['attempt'] = test_df['user_id'].map(get_user_content_attempts)


    # fill prior question had explenation
    test_df['prior_question_elapsed_time'].fillna(23916, inplace=True)

    # add harmonic mean
    test_df['hmean_user_content_accuracy'] = 2 * (
            (test_df['mean_user_accuracy'] * test_df['mean_content_accuracy']) /
            (test_df['mean_user_accuracy'] + test_df['mean_content_accuracy'])
    )

    test_df['answered_correctly'] = model.predict(test_df[used_cols])

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])

    # set previour test_df
    prev_test_df = test_df.copy()