1. 单变量分布的柱状图

   `sns.distplot(df['y'])`

2. 两个变量之间的散点图

   `df.plot.scatter(x='col', y='y')`

3. 箱型图

   `sns.boxplot(x='col', y='y'), data=df`

4. 协方差矩阵的热力图

   `corrmat = df.corr() # DataFrame` 

   `sns.heatmap(corrmat, vmax=0.8, square=True)`

   `corrmat.nlargest(10, 'y') # 取与'y'相关系数最大的10行`

   `corrmat.nlargest(10, 'y')['y'].index`

   `k = 10 # number of variables for heatmap`

   `# 找到相关系数前10的列名`

   `cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index`
   
   `cm = np.corrcoef(df_train[cols].values.T)`
   
   `sns.set(font_scale=1.25)`

   `hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)`
   
   `plt.show()`
   
   