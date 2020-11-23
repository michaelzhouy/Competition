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

state = get_state()


def get_user_data(state, test_df):
    # updated data
    attempt, mean_user_accuracy, answered_correctly_user, answered_user = [], [], [], []
    for idx, (user_id, content_id) in test_df[['user_id', 'content_id']].iterrows():
        # check if user exists
        if user_id in state:
            # check if user already answered the question, if so update it to a maximum of 4
            if content_id in state[user_id]['user_content_attempts']:
                state[user_id]['user_content_attempts'][content_id] = min(4, state[user_id]['user_content_attempts'][
                    content_id] + 1)
            # if user did not answered the question already, set the number of attempts to 0
            else:
                state[user_id]['user_content_attempts'][content_id] = 0

        # else create user with default values
        else:
            dict_keys = ['mean_user_accuracy', 'answered_correctly_user', 'answered_user', 'user_content_attempts']
            dict_default_vals = [0.680, 0, 0, dict(zip([content_id], [0]))]
            state[user_id] = dict(zip(dict_keys, dict_default_vals))

        # add user data to lists
        attempt.append(state[user_id]['user_content_attempts'][content_id])
        mean_user_accuracy.append(state[user_id]['mean_user_accuracy'])
        answered_correctly_user.append(state[user_id]['answered_correctly_user'])
        answered_user.append(state[user_id]['answered_user'])

    return attempt, mean_user_accuracy, answered_correctly_user, answered_user


# updates the user data
def update_user_data(state, features_questions_df, prev_test_df):
    for user_id, content_id, answered_correctly in prev_test_df[['user_id', 'content_id', 'answered_correctly']].values:
        # update user features
        state[user_id]['answered_correctly_user'] += answered_correctly
        state[user_id]['answered_user'] += 1
        state[user_id]['mean_user_accuracy'] = state[user_id]['answered_correctly_user'] / state[user_id]['answered_user']


env = riiideducation.make_env()
iter_test = env.iter_test()

prev_test_df = None
mean_attempt_acc_factor = FEATURES['mean_attempt_acc_factor']


for idx, (test_df, _) in tqdm(enumerate(iter_test)):
    # from 2nd iteration, update user data
    if prev_test_df is not None:
        prev_test_df['answered_correctly'] = eval(test_df['prior_group_answers_correct'].iloc[0])
        update_user_data(state, features_questions_df, prev_test_df.loc[prev_test_df['content_type_id'] == 0])
        if idx is 1:
            display(test_df)
            display(prev_test_df)

    # get user data from state and update attempt
    attempt, mean_user_accuracy, answered_correctly_user, answered_user = get_user_data(state, test_df)

    # set updated user data
    test_df['attempt'] = attempt
    test_df['mean_user_accuracy'] = mean_user_accuracy
    test_df['answered_correctly_user'] = answered_correctly_user
    test_df['answered_user'] = answered_user

    # merge with all features
    test_df = features_questions_df.merge(test_df, how='right', on='content_id')

    # fill prior question had explenation
    test_df['prior_question_elapsed_time'].fillna(23916, inplace=True)

    # add harmonic mean
    test_df['hmean_user_content_accuracy'] = 2 * (
            (test_df['mean_user_accuracy'] * test_df['mean_content_accuracy']) /
            (test_df['mean_user_accuracy'] + test_df['mean_content_accuracy'])
    )

    test_df['answered_correctly'] = model.predict(test_df[features])

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])

    # set previour test_df
    prev_test_df = test_df.copy()