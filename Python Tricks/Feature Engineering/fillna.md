1. 用众数填充缺失值

   `df['col'].fillna(df['col'].mode()[0], inplace=True)`

2. 用分组后的众数填充缺失值

   `df['col1'] = df.groupby(['col2', 'col3'])['col1'].transform(lambda x: x.fillna(x.mode()[0]))`

3. 用分组后的中位数填充缺失值

   `df['col1'] = df.groupby(['col2', 'col3'])['col1'].transform(lambda x: x.fillna(x.median()))`

4. 用每列的众数填充每列的缺失值

   `df = df.fillna(df.mode().iloc[0, :])`

