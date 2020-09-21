# 特征工程

1. 划分训练/测试集：区分每个车型、每个省份，将得到每个车型、每个省份都只有24行（24个月），将最后一个月作为测试集
2. 对popularity和salesVolume做log变换
3. 春节标记特征
- 是否是春节
- 距离春节的月份数
4. 一阶差分特征，分别对salesVolume和popularity做以下特征
- 作shift
- shift的比值特征
- diff特征
5. 二阶差分特征
- 对上一步的diff特征，进一步做diff(1)
6. 历史统计特征
- 对salesVolume和popularity的数据生成list
- 生成一个index，[1, 2, ..., 24]
- 对index计算历史前7天的的salesVolume和popularity，接着计算统计值（最大值、最小值、平均值、均值、方差、最大值-最小值）
- 对历史前7天的salesVolume和popularity计算diff，接着计算统计值（最大值、最小值、平均值、均值、方差、最大值-最小值）