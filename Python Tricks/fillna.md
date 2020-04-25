1. 用众数填充缺失值

   `df['col'].fillna(df['col'].mode()[0], inplace=True)`

2. 用分组后的众数填充缺失值

   `df['col1'] = df.groupby(['col2', 'col3'])['col1'].transform(lambda x: x.fillna(x.median()))`

3. 

