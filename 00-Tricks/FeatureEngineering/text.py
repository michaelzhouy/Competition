# -*- coding: utf-8 -*-
# @Time     : 2020/11/6 16:27
# @Author   : Michael_Zhouy
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def hashfxn(astring):
    return ord(astring[0])


def tfidf_emb(df, uid, cat_col, emb_size=20, seed=1024):
    print('Start tfidf ...')
    df[cat_col] = df[cat_col].astype(str)
    df[cat_col].fillna('-1', inplace=True)
    tmp = df.groupby(uid, as_index=False)[cat_col].agg({cat_col: list})
    tmp[cat_col] = tmp[cat_col].apply(lambda x: ' '.join(x))
    tfidf_enc = TfidfVectorizer()
    tfidf_vec = tfidf_enc.fit_transform(tmp[cat_col])
    svd_enc = TruncatedSVD(n_components=emb_size, n_iter=20, random_state=seed)
    svd_vec = svd_enc.fit_transform(tfidf_vec)
    tfidf_df = pd.DataFrame(svd_vec)
    tfidf_df.columns = ['{}_tfidf_{}'.format(cat_col, i) for i in range(emb_size)]
    res = pd.concat([tmp[[uid]], tfidf_df], axis=1)
    return res


def count2vec_emb(df, uid, cat_col, emb_size=20, seed=1024):
    print('Start count2vec ...')
    df[cat_col] = df[cat_col].astype(str)
    df[cat_col].fillna('-1', inplace=True)
    tmp = df.groupby(uid, as_index=False)[cat_col].agg({cat_col: list})
    tmp[cat_col] = tmp[cat_col].apply(lambda x: ' '.join(x))
    count_enc = CountVectorizer()
    count_vec = count_enc.fit_transform(tmp[cat_col])
    svd_enc = TruncatedSVD(n_components=emb_size, n_iter=20, random_state=seed)
    svd_vec = svd_enc.fit_transform(count_vec)
    c2v_df = pd.DataFrame(svd_vec)
    c2v_df.columns = ['{}_count2vec_{}'.format(cat_col, i) for i in range(emb_size)]
    res = pd.concat([tmp[[uid]], c2v_df], axis=1)
    return res


# workers设为1可复现训练好的词向量，但速度稍慢，若不考虑复现的话，可对此参数进行调整
def word2vec_emb(df, uid, cat_col, length):
    print('Start word2vec ...')
    tmp = df.groupby(uid, as_index=False)[cat_col].agg({cat_col: list})
    model = Word2Vec(tmp[cat_col].values, size=length, window=5, min_count=1, sg=1, hs=1,
                     workers=3, iter=10, seed=1, hashfxn=hashfxn)
    tmp[cat_col] = tmp[cat_col].apply(lambda x: pd.DataFrame([model[c] for c in x]))
    for m in range(length):
        tmp['w2v_{}_mean'.format(m)] = tmp[cat_col].apply(lambda x: x[m].mean())
    del tmp[cat_col]
    return tmp


def doc2vec_emb(df, uid, cat_col, length):
    print('Start doc2vec ...')
    tmp = df.groupby(uid, as_index=False)[cat_col].agg({cat_col: list})
    documents = [TaggedDocument(doc, [i]) for i, doc in zip(tmp[uid].values, tmp[cat_col])]
    model = Doc2Vec(documents, vector_size=length, window=5, min_count=1, workers=1, seed=1, hashfxn=hashfxn,
                    epochs=10, sg=1, hs=1)
    doc_df = tmp[uid].apply(lambda x: ','.join([str(i) for i in model[x]])).str.split(',', expand=True).apply(pd.to_numeric)
    doc_df.columns = ['{}_d2v_{}'.format(cat_col, i) for i in range(length)]
    return pd.concat([tmp[[uid]], doc_df], axis=1)
