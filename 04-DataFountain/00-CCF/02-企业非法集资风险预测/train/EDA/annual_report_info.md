1. id:企业唯一标识
2. ANCHEYEAR:年度
- [2017., 2018., 2016., 2015.]
3. STATE:状态
- [ 2.,  1., nan]
4. FUNDAM:资金数额
- 缺失率为0.747
5. MEMNUM:成员人数
- 缺失率太高，删除
6. FARNUM:农民人数
- 缺失率太高，删除
7. ANNNEWMEMNUM:本年度新增成员人数
- 缺失率太高，删除
8. ANNREDMEMNUM:本年度退出成员人数
- 缺失率太高，删除
9. EMPNUM:从业人数
10. EMPNUMSIGN:从业人数是否公示
- [nan,  2.,  1.]
11. BUSSTNAME:经营状态名称
- [nan, '开业', '歇业', '停业', '清算']
12. COLGRANUM:其中高校毕业生人数经营者
13. RETSOLNUM:其中退役士兵人数经营者
14. DISPERNUM:其中残疾人人数经营者
15. UNENUM:其中下岗失业人数经营者
16. COLEMPLNUM:其中高校毕业生人数雇员
17. RETEMPLNUM:其中退役士兵人数雇员
18. DISEMPLNUM:其中残疾人人数雇员
19. UNEEMPLNUM:其中下岗失业人数雇员
20. WEBSITSIGN:是否有网站标志
21. FORINVESTSIGN:是否有对外投资企业标志
22. STOCKTRANSIGN:有限责任公司本年度是否发生股东股权转让标志
23. PUBSTATE:公示状态：1 全部公示，2部分公示,3全部不公示]


二值特征
['STATE', 'EMPNUMSIGN', 'WEBSITSIGN', 'FORINVESTSIGN', 'STOCKTRANSIGN']

三值特征
['PUBSTATE']

四值特征
['ANCHEYEAR', 'BUSSTNAME']

num_cols = ['FUNDAM', 'EMPNUM', 'COLGRANUM', 'RETSOLNUM', 'DISPERNUM', 'COLEMPLNUM', 'RETEMPLNUM', 'DISEMPLNUM', 'UNEEMPLNUM']

null_cols = ['MEMNUM', 'FARNUM', 'ANNNEWMEMNUM', 'ANNREDMEMNUM']