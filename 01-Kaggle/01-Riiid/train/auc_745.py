%%time
cols_to_load = ['row_id', 'user_id', 'answered_correctly', 'content_id', 'prior_question_had_explanation']
train = pd.read_pickle("../input/riiid-train-data-multiple-formats/riiid_train.pkl.gzip")[cols_to_load]
train['prior_question_had_explanation'] = train['prior_question_had_explanation'].astype('bool')

print("Train size:", train.shape)


%%time

questions = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/questions.csv')
lectures = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/lectures.csv')
example_test = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/example_test.csv')
example_sample_submission = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/example_sample_submission.csv')

%%time
#adding user features
user_df = train[train.answered_correctly != -1].groupby('user_id').agg({'answered_correctly': ['count', 'mean']}).reset_index()
user_df.columns = ['user_id', 'user_questions', 'user_mean']

user_lect = train.groupby(["user_id", "answered_correctly"]).size().unstack()
user_lect.columns = ['Lecture', 'Wrong', 'Right']
user_lect['Lecture'] = user_lect['Lecture'].fillna(0)
user_lect = user_lect.astype('Int64')
user_lect['watches_lecture'] = np.where(user_lect.Lecture > 0, 1, 0)
user_lect = user_lect.reset_index()
user_lect = user_lect[['user_id', 'watches_lecture']]

user_df = user_df.merge(user_lect, on = "user_id", how = "left")
del user_lect
user_df.head()

%%time
#adding content features
content_df = train[train.answered_correctly != -1].groupby('content_id').agg({'answered_correctly': ['count', 'mean']}).reset_index()
content_df.columns = ['content_id', 'content_questions', 'content_mean']

%%time
#using one of the validation sets composed by tito
cv2_train = pd.read_pickle("../input/riiid-cross-validation-files/cv2_train.pickle")['row_id']
cv2_valid = pd.read_pickle("../input/riiid-cross-validation-files/cv2_valid.pickle")['row_id']

train = train[train.answered_correctly != -1]
validation = train[train.row_id.isin(cv2_valid)]
train = train[train.row_id.isin(cv2_train)]

validation = validation.drop(columns = "row_id")
train = train.drop(columns = "row_id")

del cv2_train, cv2_valid
gc.collect()

def merge_fill_na(df):
    df = df.merge(user_df, on = "user_id", how = "left")
    df = df.merge(content_df, on = "content_id", how = "left")
    df['content_questions'].fillna(0, inplace = True)
    df['content_mean'].fillna(0.5, inplace = True)
    df['watches_lecture'].fillna(0, inplace = True)
    df['user_questions'].fillna(0, inplace = True)
    df['user_mean'].fillna(0.5, inplace = True)
    df[['content_questions', 'user_questions']] = df[['content_questions', 'user_questions']].astype(int)
    return(df)

train = merge_fill_na(train)
train.sample(5)

validation = merge_fill_na(validation)
validation.sample(5)

features = ['user_questions', 'user_mean', 'content_questions', 'content_mean', 'watches_lecture']

#for now just taking 10.000.000 rows for training
train = train.sample(n=10000000, random_state = 1)

y_train = train['answered_correctly']
train = train[features]

y_val = validation['answered_correctly']
validation = validation[features]

params = {'objective': 'binary',
          'metric': 'auc',
          'seed': 2020,
          'learning_rate': 0.1, #default
          "boosting_type": "gbdt" #default
         }

lgb_train = lgb.Dataset(train, y_train, categorical_feature = ['watches_lecture'])
lgb_eval = lgb.Dataset(validation, y_val, categorical_feature = ['watches_lecture'])
del train, y_train, validation, y_val
gc.collect()

%%time
model = lgb.train(
    params, lgb_train,
    valid_sets=[lgb_train, lgb_eval],
    verbose_eval=50,
    num_boost_round=10000,
    early_stopping_rounds=8
)

lgb.plot_importance(model)
plt.show()

env = riiideducation.make_env()
iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:
    test_df = merge_fill_na(test_df)
    test_df['answered_correctly'] =  model.predict(test_df[features])
    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])