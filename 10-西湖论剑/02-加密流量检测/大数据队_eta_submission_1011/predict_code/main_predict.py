# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def test_func(test_path,save_path):
	# 请填写测试代码
	test = pd.read_csv(test_path)
	# 选手不得改变格式，测试代码跑不通分数以零算

	# #####选手填写测试集处理逻辑,在指定文件夹下生成可提交的csv文件

	# demo#
	submission = test[['eventId']]
	submission['label'] = 0
	submission.to_csv(save_path + '大数据队_eta_submission_1011.csv',index = False,encoding='utf-8')


if __name__ == '__main__':
	test_path = '../data/test_1.csv'
	sava_path = '../results/'
	test_func(test_path,sava_path)
