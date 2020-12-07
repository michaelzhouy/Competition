# -*- coding:utf-8 -*-
# Time   : 2020/12/7 22:52
# Email  : 15602409303@163.com
# Author : Zhou Yang

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Input, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping

# 读取测数据集 验证集可自行划分
train = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

train_index = int(0.8 * len(train))
train_df = train.iloc[:train_index, :]
val_df = train.iloc[train_index:, :]

train_df.columns = ['review', 'rating']
val_df.columns = ['review', 'rating']

train_y = train_df.rating
val_y = val_df.rating


le = LabelEncoder()
train_y = le.fit_transform(train_y).reshape(-1, 1)
val_y = le.transform(val_y).reshape(-1, 1)


ohe = OneHotEncoder()
train_y = ohe.fit_transform(train_y).toarray()
val_y = ohe.transform(val_y).toarray()


max_words = 5000
max_len = 128
tok = Tokenizer(num_words=max_words)  # 使用的最大词语数为5000
tok.fit_on_texts(train_df.review)


# 使用word_index属性可以看到每次词对应的编码
# 使用word_counts属性可以看到每个词对应的频数
for ii, iterm in enumerate(tok.word_index.items()):
    if ii < 10:
        print(iterm)
    else:
        break
print("===================")
for ii, iterm in enumerate(tok.word_counts.items()):
    if ii < 10:
        print(iterm)
    else:
        break


# 对每个词编码之后，每句中的每个词就可以用对应的编码表示，即每条可以转变成一个向量了：
train_seq = tok.texts_to_sequences(train_df.review)
val_seq = tok.texts_to_sequences(val_df.review)
test_seq = tok.texts_to_sequences(test_df.review)

# 将每个序列调整为相同的长度
train_seq_mat = sequence.pad_sequences(train_seq, maxlen=max_len)
val_seq_mat = sequence.pad_sequences(val_seq, maxlen=max_len)
test_seq_mat = sequence.pad_sequences(test_seq, maxlen=max_len)

print(train_seq_mat.shape)
print(val_seq_mat.shape)
# print(test_seq_mat.shape)


# 定义LSTM模型
inputs = Input(name='inputs', shape=[max_len])
# Embedding(词汇表大小,batch大小,每个词长)
layer = Embedding(max_words+1, 128, input_length=max_len)(inputs)
layer = LSTM(128)(layer)
layer = Dense(128, activation="relu", name="FC1")(layer)
layer = Dropout(0.5)(layer)
layer = Dense(5, activation="softmax", name="FC2")(layer)
model = Model(inputs=inputs, outputs=layer)
print(model.summary())
model.compile(loss="categorical_crossentropy", optimizer='Adam', metrics=["accuracy"])

model_fit = model.fit(
    train_seq_mat, train_y,
    batch_size=4,
    epochs=4,
    validation_data=(val_seq_mat, val_y),
    callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.001)]  # 当val-loss不再提升时停止训练
)

test_pre = model.predict(test_seq_mat)

# ## 评价预测效果，计算混淆矩阵
# confm = metrics.confusion_matrix(np.argmax(test_pre,axis=1),np.argmax(test_y,axis=1))
# ## 混淆矩阵可视化

# print(metrics.classification_report(np.argmax(test_pre,axis=1),np.argmax(test_y,axis=1)))

train_pre = model.predict(train_seq_mat)

test_pre = model.predict(test_seq_mat)
test_pre = np.argmax(test_pre, axis=1)
test_pre = test_pre.tolist()
test_pre = pd.DataFrame(test_pre)
test_pre = test_pre + 1
test_pre.to_csv('../sub/sub1121.csv', header=False)

print(test_pre)
