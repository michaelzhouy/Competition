import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

N_ROWS = 99271300


def get_index_np():
    return np.arange(N_ROWS) # speeds up dataframe creation


data_types_dict = {
    'user_id': 'int32',
    'content_id': 'int16',
    'answered_correctly': 'int8',
    'prior_question_elapsed_time': 'float32',
    'prior_question_had_explanation': 'bool'
}
train_df = dt.fread('../input/riiid-test-answer-prediction/train.csv', columns=set(data_types_dict.keys())).to_pandas()

# 课程数据删除
train_df = train_df[train_df[target] != -1].reset_index(drop=True)
train_df['prior_question_had_explanation'].fillna(False, inplace=True)
train_df = train_df.astype(data_types_dict)

questions_df = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv')
questions_df = questions_df[['question_id', 'part']]
questions_df.rename(columns={'questions_id': 'content_id'}, inplace=True)
train_df = train_df.merge(questions_df, on='content_id', how='left')
del questions_df
gc.collect()

content_mean = train_df.groupby('content_id')['answered_correctly'].agg(['mean'])

# 2020-11-22
def get_state():
    # create DataFrame of features
    features_df = pd.DataFrame(index=get_index_np())
    for col in tqdm(['user_id', 'content_id', 'answered_correctly']):
        features_df[col] = train_df[col]

    # fill dictionary with default values
    state = dict()
    for user_id in features_df['user_id'].unique():
        state[user_id] = {}
    total = len(state.keys())

    # add user content attempts
    user_content = features_df.groupby('user_id')['content_id'].apply(np.array).apply(np.sort).apply(np.unique)
    user_attempts = features_df.groupby(['user_id', 'content_id'])['content_id'].count().astype(np.uint8).groupby('user_id').apply(np.array).values
    user_attempts -= 1

    for user_id, content, attempt in tqdm(zip(state.keys(), user_content, user_attempts), total=total):
        state[user_id]['user_content_attempts'] = dict(zip(content, attempt))

    del user_content, user_attempts
    gc.collect()

    
    # compute user features over all train data
    mean_user_accuracy = features_df.groupby('user_id')['answered_correctly'].mean().values
    answered_correctly_user = features_df.groupby('user_id')['answered_correctly'].sum().values
    answered_user = features_df.groupby('user_id')['answered_correctly'].count().values
    
    for idx, user_id in enumerate(state.keys()):
        state[user_id]['mean_user_accuracy'] = mean_user_accuracy[idx]
        state[user_id]['answered_correctly_user'] = answered_correctly_user[idx]
        state[user_id]['answered_user'] = answered_user[idx]

    return state