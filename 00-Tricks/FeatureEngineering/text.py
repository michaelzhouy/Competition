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


def tfidf(input_values, output_num, output_prefix, seed=1024):
    tfidf_enc = TfidfVectorizer()
    tfidf_vec = tfidf_enc.fit_transform(input_values)
    svd_tmp = TruncatedSVD(n_components=output_num, n_iter=20, random_state=seed)
    svd_tmp = svd_tmp.fit_transform(tfidf_vec)
    svd_tmp = pd.DataFrame(svd_tmp)
    svd_tmp.columns = ['{}_tfidf_{}'.format(output_prefix, i) for i in range(output_num)]
    return svd_tmp


def count2vec(input_values, output_num, output_prefix, seed=1024):
    count_enc = CountVectorizer()
    count_vec = count_enc.fit_transform(input_values)
    svd_tmp = TruncatedSVD(n_components=output_num, n_iter=20, random_state=seed)
    svd_tmp = svd_tmp.fit_transform(count_vec)
    svd_tmp = pd.DataFrame(svd_tmp)
    svd_tmp.columns = ['{}_countvec_{}'.format(output_prefix, i) for i in range(output_num)]
    return svd_tmp


def get_geohash_tfidf(df, group_id, group_target, num):
    # tfidf_df = get_geohash_tfidf(df, 'ID', 'lat_lon', 30)
    df[group_target] = df.apply(lambda x: geohash_encode(x['lat'], x['lon'], 7), axis=1)
    tmp = df.groupby(group_id)[group_target].agg(list).reset_index()
    tmp[group_target] = tmp[group_target].apply(lambda x: ' '.join(x))

    tfidf_tmp = tfidf(tmp[group_target], num, group_target)
    count_tmp = count2vec(tmp[group_target], num, group_target)
    return pd.concat([tmp[[group_id]], tfidf_tmp, count_tmp], axis=1)


def get_grad_tfidf(df, group_id, group_target, num):
    # grad_tfidf = get_grad_tfidf(df, 'ID', 'grad', 30)
    grad_df = df.groupby(group_id)['lat'].apply(lambda x: np.gradient(x)).reset_index()
    grad_df['lon'] = df.groupby(group_id)['lon'].apply(lambda x: np.gradient(x))
    grad_df['lat'] = grad_df['lat'].apply(lambda x: np.round(x, 4))
    grad_df['lon'] = grad_df['lon'].apply(lambda x: np.round(x, 4))
    grad_df[group_target] = grad_df.apply(lambda x: str(x['lat']) + ' ' + str(x['lon']), axis=1)

    tfidf_tmp = tfidf(grad_df[group_target], num, group_target)
    return pd.concat([grad_df[[group_id]], tfidf_tmp], axis=1)


def get_sample_tfidf(df, group_id, group_target, num):
    # sample_tfidf = get_sample_tfidf(df, 'ID', 'sample', 30)
    tmp = df.groupby(group_id)['lat_lon'].apply(lambda x: x.sample(frac=0.1, random_state=1)).reset_index()
    del tmp['level_1']
    tmp.columns = [group_id, group_target]
    tmp = tmp.groupby(group_id)[group_target].agg(list).reset_index()
    tmp[group_target] = tmp[group_target].apply(lambda x: ' '.join(x))

    tfidf_tmp = tfidf(tmp[group_target], num, group_target)
    return pd.concat([tmp[[group_id]], tfidf_tmp], axis=1)


# workers设为1可复现训练好的词向量，但速度稍慢，若不考虑复现的话，可对此参数进行调整
def w2v_feat(df, group_id, feat, length):
    # w2v_df = w2v_feat(df, 'ID', 'lat_lon', 30)
    print('start word2vec ...')
    data_frame = df.groupby(group_id)[feat].agg(list).reset_index()
    model = Word2Vec(data_frame[feat].values, size=length, window=5, min_count=1, sg=1, hs=1,
                     workers=1, iter=10, seed=1, hashfxn=hashfxn)
    data_frame[feat] = data_frame[feat].apply(lambda x: pd.DataFrame([model[c] for c in x]))
    for m in range(length):
        data_frame['w2v_{}_mean'.format(m)] = data_frame[feat].apply(lambda x: x[m].mean())
    del data_frame[feat]
    return data_frame


def d2v_feat(df, group_id, feat, length):
    print('start doc2vec ...')
    data_frame = df.groupby(group_id)[feat].agg(list).reset_index()
    documents = [TaggedDocument(doc, [i]) for i, doc in zip(data_frame[group_id].values, data_frame[feat])]
    model = Doc2Vec(documents, vector_size=length, window=5, min_count=1, workers=1, seed=1, hashfxn=hashfxn,
                    epochs=10, sg=1, hs=1)
    doc_df = data_frame[group_id].apply(lambda x: ','.join([str(i) for i in model[x]])).str.split(',', expand=True).apply(pd.to_numeric)
    doc_df.columns = ['{}_d2v_{}'.format(feat, i) for i in range(length)]
    return pd.concat([data_frame[[group_id]], doc_df], axis=1)
