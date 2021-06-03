# 特征工程
2. 计算seq(时间序列标记)
2. 组合：model+adcode，model_adcode_seq，adcode_seq，model_seq
3. shift特征：根据model_adcode_seq（唯一），对label做shift操作
4. 分组shift特征：分别对adcode_seq和model_seq做groupby求和，生成一个只有adcode_seq和label之和的两列的DataFrame，最后对该DataFrame对象做shift操作
5. 历史销量特征：对shift_label_{}，shift_popularity_{}，adcode_seq_shift_{}和adcode_seq_shift_{}，计算前2、3、4、6、12个月的销量和与销量均值

# 建模
1. 分车型单独预测
2. 区分数值特征和类别特征
3. 只用一年内的数据做训练
4. 将预测结果合并到训练集中，一起训练
5. 将所有预测结果合并到一起，用作提交


LGB_Model_2特征工程与此相同，预测时不分车型预测