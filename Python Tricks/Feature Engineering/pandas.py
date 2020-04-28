# 数据存储
df.to_hdf('data.h5', 'df')
pd.read_hdf('data.h5')

# 筛选object特征
df_object = df.select_dtypes(include=['object'])
df_numerical = df.select_dtypes(exclude=['object'])

import warnings
warnings.filterwarnings('ignore')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)

# 读取文件下所有的文件
def read_data(path):
	data_list = []
	for f in os.listdir(path):
		print(f)
		df = pd.read_csv(path + os.sep + f)
		print(df.shape)
		data_list.append(df)
		del df
		gc.collect()
	
	res = pd.concat(data_list, ignore_index=True)
	return res
