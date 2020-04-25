## count_encoder

`def count_coding(df, fea_col):
    for f in fea_col:
        df[f + '_count'] = df[f].map(df[f].value_counts())
    return df`

