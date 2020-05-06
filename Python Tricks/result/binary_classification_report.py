# -*- coding: utf-8 -*-
# @Time     : 2020/5/6 20:24
# @Author   : Michael_Zhouy

import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)


def binary_classification_report(y_true, y_pred, qlist=None, cum=True):
    """

    Parameters
    ----------
    y_true
    y_pred
    qlist
    cum

    Returns
    -------

    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    assert len(y_pred) == len(y_true)
    df = pd.DataFrame({'pred': np.ravel(y_pred),
                       'true': np.ravel(y_true)})
    df.sort_values('pred', ascending=False, inplace=True)
    if qlist is None:
        qlist = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
                 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                 0.96, 0.97, 0.98, 0.99, 1]

    qlist = np.sort(qlist)
    if qlist[0] < 0 or qlist[-1] > 1:
        print('qlist should be in range [0, 1]')
        return None

    total_cnt = df.shape[0]
    total_pos_cnt = df['true'].sum()
    total_neg_pct = total_pos_cnt / total_cnt

    df['depth'] = qlist[np.searchsorted(qlist, np.arange(1, total_cnt + 1) / total_cnt)]

    def agg_func(grouped):
        pos_count = grouped['true'].sum()
        neg_count = grouped.shape[0] - pos_count
        all_count = pos_count + neg_count

        s = pd.Series({'pos_count': pos_count,
                       'neg_count': neg_count,
                       'all_count': all_count})
        return s

    report = df.groupby('depth').apply(agg_func)
    if cum:
        report = report.apply(np.cumsum)
    else:
        report = report.apply(np.sum)

    report.reset_index(inplace=True)
    report['hit_rate'] = report['pos_count'] / report['all_count']
    report['coverage'] = report['pos_count'] / total_pos_cnt
    report['lift_rate'] = report['hit_rate'] / total_neg_pct
    proba_idx = np.floor(report['depth'] * total_cnt).astype('int') - 1
    proba_idx[proba_idx < 0] = 0
    report['prob'] = df['pred'].values[proba_idx]
