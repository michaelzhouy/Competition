import numpy as np
import pandas as pd

import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score
from multiprocessing import cpu_count
from tqdm.notebook import tqdm

# these imports are used to convert the tree to PNG
from cairosvg import svg2png
from PIL import Image
from io import BytesIO

import gc
import os
import sys

VERSION = 'V1E'
NUM_BOOST_ROUND = 2500
VERBOSE_EVAL = 10
METRICS = ['auc']
N_ROWS = 99271300


def get_index_np():
    return np.arange(N_ROWS) # speeds up dataframe creation


# features are saved as compressed numpy arrays, much more efficient than a pandas DataFrame!
FEATURES = np.load(f'/kaggle/input/riiid-answer-correctness-prediction-features/train_features_{VERSION}.npz', allow_pickle=True)

given_features = ['prior_question_elapsed_time']

deduced_features = [
    # user features
    'mean_user_accuracy',
    'answered_correctly_user',
    'answered_user',
    # content features
    'mean_content_accuracy',
    # part features
    'part',
    # other features
    'hmean_user_content_accuracy',
    'attempt',
]

features = given_features + deduced_features

target = 'answered_correctly'

# add categorical features indices
categorical_feature = ['part', 'tags', 'tags_label', 'prior_question_had_explanation']
categorical_feature_idxs = []
for v in categorical_feature:
    try:
        categorical_feature_idxs.append(features.index(v))
    except:
        pass


def get_train_val_idxs(TRAIN_SIZE, VAL_SIZE):
    train_idxs = []
    val_idxs = []
    NEW_USER_FRAC = 1 / 4  # fraction of new users in
    np.random.seed(42)

    # create df with user_ids and indices
    df = pd.DataFrame(index=get_index_np())
    for col in ['user_id']:
        df[col] = FEATURES[col]

    df['index'] = df.index.values.astype(np.uint32)
    user_id_index = df.groupby('user_id')['index'].apply(np.array)

    # iterate over users in random order
    for indices in user_id_index.sample(user_id_index.size, random_state=42):
        if len(train_idxs) > TRAIN_SIZE:
            break

        # fill validation data
        if len(val_idxs) < VAL_SIZE:
            # add new user
            if np.random.rand() < NEW_USER_FRAC:
                val_idxs += list(indices)
            # randomly split user between train and val otherwise
            else:
                offset = np.random.randint(0, indices.size)
                train_idxs += list(indices[:offset])
                val_idxs += list(indices[offset:])
        else:
            train_idxs += list(indices)

    return train_idxs, val_idxs


train_idxs, val_idxs = get_train_val_idxs(int(50e6), 2.5e6)
print(f'len train_idxs: {len(train_idxs)}, len validation_idxs: {len(val_idxs)}')


def make_x_y(FEATURES, train_idxs, val_idxs):
    # create numpy arrays
    X_train = np.ndarray(shape=(len(train_idxs), len(features)), dtype=np.float32)
    X_val = np.ndarray(shape=(len(val_idxs), len(features)), dtype=np.float32)

    # now fill them up
    for idx, feature in enumerate(tqdm(features)):
        X_train[:, idx] = FEATURES[feature][train_idxs].astype(np.float32)
        X_val[:, idx] = FEATURES[feature][val_idxs].astype(np.float32)

    # add the target
    y_train = FEATURES[target][train_idxs].astype(np.int8)
    y_val = FEATURES[target][val_idxs].astype(np.int8)

    return X_train, y_train, X_val, y_val


X_train, y_train, X_val, y_val = make_x_y(FEATURES, train_idxs, val_idxs)

print(f'X_train.shape: {X_train.shape}\t y_train.shape: {y_train.shape}')
print(f'X_val.shape: {X_val.shape}\t y_val.shape: {y_val.shape}')


# show train features
pd.DataFrame(X_train[:10], columns=features)

y_train[:10]

# make train and validation dataset
train_data = lgb.Dataset(
    data = X_train,
    label = y_train,
    categorical_feature = None,
)

val_data = lgb.Dataset(
    data = X_val,
    label = y_val,
    categorical_feature = None,
)

del X_train, y_train, X_val, y_val
gc.collect()

# NEW from:
lgbm_params = {
    'objective': 'binary',
    'metric': METRICS,
}


def train():
    evals_result = {}
    model = lgb.train(
        params=lgbm_params,
        train_set=train_data,
        valid_sets=[val_data],
        num_boost_round=NUM_BOOST_ROUND,
        verbose_eval=VERBOSE_EVAL,
        evals_result=evals_result,
        early_stopping_rounds=10,
        categorical_feature=categorical_feature_idxs,
        feature_name=features,
    )

    # save model
    model.save_model(f'model_{VERSION}_{NUM_BOOST_ROUND}.lgb')

    return model, evals_result


model, evals_result = train()


def plot_history(evals_result):
    for metric in METRICS:
        plt.figure(figsize=(20, 8))

        for key in evals_result.keys():
            history_len = len(evals_result.get(key)[metric])
            history = evals_result.get(key)[metric]
            x_axis = np.arange(1, history_len + 1)
            plt.plot(x_axis, history, label=key)

        x_ticks = list(filter(lambda e: (e % (history_len // 100 * 10) == 0) or e == 1, x_axis))
        plt.xticks(x_ticks, fontsize=12)
        plt.yticks(fontsize=12)

        plt.title(f'{metric.upper()} History of training', fontsize=18);
        plt.xlabel('EPOCH', fontsize=16)
        plt.ylabel(metric.upper(), fontsize=16)

        if metric in ['auc']:
            plt.legend(loc='upper left', fontsize=14)
        else:
            plt.legend(loc='upper right', fontsize=14)
        plt.grid()
        plt.show()


plot_history(evals_result)


# plot the feature importance in terms of gain and split
def show_feature_importances(model, importance_type, max_num_features=10 ** 10):
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = features
    feature_importances['value'] = pd.DataFrame(model.feature_importance(importance_type))
    feature_importances = feature_importances.sort_values(by='value', ascending=False)  # sort feature importance
    feature_importances.to_csv(f'feature_importances_{importance_type}.csv')  # write feature importance to csv
    feature_importances = feature_importances[:max_num_features]  # only show max_num_features

    plt.figure(figsize=(20, 8))
    plt.xlim([0, feature_importances.value.max() * 1.1])
    plt.title(f'Feature {importance_type}', fontsize=18);
    sns.barplot(data=feature_importances, x='value', y='feature', palette='rocket');
    for idx, v in enumerate(feature_importances.value):
        plt.text(v, idx, "  {:.2e}".format(v))


show_feature_importances(model, 'gain')
show_feature_importances(model, 'split')


# show tree and save as png
def save_tree_diagraph(model):
    tree_digraph = lgb.create_tree_digraph(model, show_info=['split_gain', 'internal_count'])

    tree_png = svg2png(tree_digraph._repr_svg_(), output_width=3840)
    tree_png = Image.open(BytesIO(tree_png))

    tree_png.save('create_tree_digraph.png')

    display(tree_png)


save_tree_diagraph(model)

# remove train and validation data to free memory before prediction phase
del train_data
gc.collect()


# dataframe with question features used for merging with test_df
def get_features_questions_df():
    # create DataFrame of features
    features_df = pd.DataFrame(index=get_index_np())
    for col in tqdm(['content_id', 'part', 'tags', 'tags_label', 'mean_content_accuracy']):
        features_df[col] = FEATURES[col]

    # content features
    features_questions_df = features_df.groupby('content_id')[[
        # merge keys
        'content_id',
        'part',
        'tags',
        'tags_label',
        # content
        'mean_content_accuracy',
    ]].first().reset_index(drop=True).sort_values('content_id')

    return features_questions_df


features_questions_df = get_features_questions_df()
print(f'features_questions_df, rows: {features_questions_df.shape[0]}')
display(features_questions_df.head())


def get_state():
    # create DataFrame of features
    features_df = pd.DataFrame(index=get_index_np())
    for col in tqdm(['user_id', 'content_id', 'answered_correctly']):
        features_df[col] = FEATURES[col]

    # compute user features over all train data
    mean_user_accuracy = features_df.groupby('user_id')['answered_correctly'].mean().values
    answered_correctly_user = features_df.groupby('user_id')['answered_correctly'].sum().values
    answered_user = features_df.groupby('user_id')['answered_correctly'].count().values

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

    for idx, user_id in enumerate(state.keys()):
        state[user_id]['mean_user_accuracy'] = mean_user_accuracy[idx]
        state[user_id]['answered_correctly_user'] = answered_correctly_user[idx]
        state[user_id]['answered_user'] = answered_user[idx]

    return state


state = get_state()
print('Example of the state for user 2746, attempt counting starts at 0 as the pandas cumcount function is used to create the attempt feature')
display(state[2746])


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

import riiideducation

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

submission = pd.read_csv('./submission.csv')
submission.info()
submission.head()