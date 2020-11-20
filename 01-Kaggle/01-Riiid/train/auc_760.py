import pandas as pd
import numpy as np
import gc
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm.notebook import tqdm
import lightgbm as lgb

train_pickle = '../input/riiid-cross-validation-files/cv1_train.pickle'
valid_pickle = '../input/riiid-cross-validation-files/cv1_valid.pickle'
question_file = '../input/riiid-test-answer-prediction/questions.csv'
debug = False
validaten_flg = False


def add_user_feats(df, answered_correctly_sum_u_dict, count_u_dict):
    acsu = np.zeros(len(df), dtype=np.int32)
    cu = np.zeros(len(df), dtype=np.int32)
    for cnt, row in enumerate(tqdm(df[['user_id', 'answered_correctly']].values)):
        acsu[cnt] = answered_correctly_sum_u_dict[row[0]]  # 回答正确的字典
        cu[cnt] = count_u_dict[row[0]]  # 回答正确的字典
        answered_correctly_sum_u_dict[row[0]] += row[1]  # 开始计数，回答正确的数
        count_u_dict[row[0]] += 1  # 总共做的题目
    user_feats_df = pd.DataFrame({'answered_correctly_sum_u': acsu, 'count_u': cu})
    user_feats_df['answered_correctly_avg_u'] = user_feats_df['answered_correctly_sum_u'] / user_feats_df['count_u']
    # 回答正确的比例
    df = pd.concat([df, user_feats_df], axis=1)
    return df


def add_user_feats_without_update(df, answered_correctly_sum_u_dict, count_u_dict):
    acsu = np.zeros(len(df), dtype=np.int32)
    cu = np.zeros(len(df), dtype=np.int32)
    for cnt, row in enumerate(df[['user_id']].values):
        acsu[cnt] = answered_correctly_sum_u_dict[row[0]]
        cu[cnt] = count_u_dict[row[0]]
    user_feats_df = pd.DataFrame({'answered_correctly_sum_u': acsu, 'count_u': cu})
    user_feats_df['answered_correctly_avg_u'] = user_feats_df['answered_correctly_sum_u'] / user_feats_df['count_u']
    df = pd.concat([df, user_feats_df], axis=1)
    return df


def update_user_feats(df, answered_correctly_sum_u_dict, count_u_dict):
    for row in df[['user_id', 'answered_correctly', 'content_type_id']].values:
        if row[2] == 0:
            answered_correctly_sum_u_dict[row[0]] += row[1]
            count_u_dict[row[0]] += 1


feld_needed = ['row_id', 'user_id', 'content_id', 'content_type_id', 'answered_correctly',
               'prior_question_elapsed_time', 'prior_question_had_explanation']
train = pd.read_pickle(train_pickle)[feld_needed]
valid = pd.read_pickle(valid_pickle)[feld_needed]
if debug:
    train = train[:1000000]
    valid = valid[:10000]
train = train.loc[train.content_type_id == False].reset_index(drop=True)
valid = valid.loc[valid.content_type_id == False].reset_index(drop=True)

# answered correctly average for each content
content_df = train[['content_id', 'answered_correctly']].groupby(['content_id']).agg(['mean']).reset_index()
content_df.columns = ['content_id', 'answered_correctly_avg_c']
train = pd.merge(train, content_df, on=['content_id'], how="left")
valid = pd.merge(valid, content_df, on=['content_id'], how="left")

# user stats features with loops
answered_correctly_sum_u_dict = defaultdict(int)
count_u_dict = defaultdict(int)
train = add_user_feats(train, answered_correctly_sum_u_dict, count_u_dict)
valid = add_user_feats(valid, answered_correctly_sum_u_dict, count_u_dict)

# fill with mean value for prior_question_elapsed_time
# note that `train.prior_question_elapsed_time.mean()` dose not work!
# please refer https://www.kaggle.com/its7171/can-we-trust-pandas-mean for detail.
prior_question_elapsed_time_mean = train.prior_question_elapsed_time.dropna().values.mean()
train['prior_question_elapsed_time_mean'] = train.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)
valid['prior_question_elapsed_time_mean'] = valid.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)

# use only last 30M training data for limited memory on kaggle env.
# train = train[-30000000:]

# part
questions_df = pd.read_csv(question_file)
train = pd.merge(train, questions_df[['question_id', 'part']], left_on='content_id', right_on='question_id', how='left')
valid = pd.merge(valid, questions_df[['question_id', 'part']], left_on='content_id', right_on='question_id', how='left')

# changing dtype to avoid lightgbm error
train['prior_question_had_explanation'] = train.prior_question_had_explanation.fillna(False).astype('int8')
valid['prior_question_had_explanation'] = valid.prior_question_had_explanation.fillna(False).astype('int8')

lgb_train = lgb.Dataset(train[FEATS], y_tr)
lgb_valid = lgb.Dataset(valid[FEATS], y_va)
del train, y_tr
_ = gc.collect()

model = lgb.train(
    {'objective': 'binary'},
    lgb_train,
    valid_sets=[lgb_train, lgb_valid],
    verbose_eval=100,
    num_boost_round=10000,
    early_stopping_rounds=10
)
print('auc:', roc_auc_score(y_va, model.predict(valid[FEATS])))
_ = lgb.plot_importance(model)

env = riiideducation.make_env()
iter_test = env.iter_test()
# model = lgb.Booster(model_file='../input/riiid-lgbm-starter/model.txt')

previous_test_df = None
for (test_df, sample_prediction_df) in iter_test:
    if previous_test_df is not None:
        previous_test_df[TARGET] = eval(test_df["prior_group_answers_correct"].iloc[0])
        update_user_feats(previous_test_df, answered_correctly_sum_u_dict, count_u_dict)

    previous_test_df = test_df.copy()
    test_df = test_df[test_df['content_type_id'] == 0].reset_index(drop=True)
    test_df = add_user_feats_without_update(test_df, answered_correctly_sum_u_dict, count_u_dict)
    test_df = pd.merge(test_df, content_df, on='content_id', how="left")
    test_df = pd.merge(test_df, questions_df, left_on='content_id', right_on='question_id', how='left')
    test_df['prior_question_had_explanation'] = test_df.prior_question_had_explanation.fillna(False).astype('int8')
    test_df['prior_question_elapsed_time_mean'] = test_df.prior_question_elapsed_time.fillna(
        prior_question_elapsed_time_mean)

    test_df[TARGET] = model.predict(test_df[FEATS])
    set_predict(test_df[['row_id', TARGET]])