# -*- coding: utf-8 -*-
# @Time     : 2020/6/14 12:16
# @Author   : Michael_Zhouy

gb = df.groupby(['user_id', 'page_id'], ax_index=False).agg(
    {'ad_price': {'max_price': np.max, 'min_price': np.min}})

gb.columns = ['user_id', 'page_id', 'min_price', 'max_price']

df = pd.merge(df, gb, on=['user_id', 'page_id'], how='left')
