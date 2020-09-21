## get_shift_feature
1. group + shift数值
2. map过来

## get_adjoin_feature
1. 求和：添加一列，全零，遍历求和
2. 求均值
3. 求差值：(当前$shift_i$)-($shift_{space+i}$)
4. 求比例：(当前$shift_i$)/($shift_{space+i}$)

## get_series_feature
1. 添加一列，全零，用于求和
2. 通过遍历将shift特征添加到一个list中去
3. 通过apply按行求和、求平均、均值、最大最小、标准差、最大-最小