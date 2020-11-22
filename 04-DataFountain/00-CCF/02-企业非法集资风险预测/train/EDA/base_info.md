1. id:企业唯一标识
2. oplocdistrict:行政区划代码
- 16个值
3. industryphy:行业类别代码
- 20个值
4. industryco:行业细类代码
- 346个值
5. dom:经营地址
- 取值很多，考虑其他编码
6. opscope:经营范围
- 中文考虑其他编码
7. enttype:企业类型
- 17个值
8. enttypeitem:企业类型小类
- 32个值
9. opfrom:经营期限起
10. opto:经营期限止
11. state:状态
- 6个值
12. orgid:机构标识
- 78个值
13. jobid:职位标识
- 434个值
14. adbusign:是否广告经营
- 2值
15. townsign:是否城镇
- 2值
16. regtype:主题登记类型
- 3值
17. empnum:从业人数，
- 32个值
18. compform:组织形式
19. parnum:合伙人数，
- 52个值
20. exenum:执行人数
- 51个值
21. opform:经营方式
- 34个值
22. ptbusscope:兼营范围
- 全空
23. venind:风险行业
24. enttypeminu:企业类型细类
- 27个值
25. midpreindcode:中西部优势产业代码
- 全空
26. protype:项目类型
27. oploc:经营场所
- 5351个值
28. regcap:注册资本（金）
- 1144个值
29. reccap:实缴资本，
- 598个值
30. forreccap:实缴资本（外方）
- 12个值
31. forregcap:注册资本（外方）
- 39个值
32. congro:投资总额
- 34个值
33. enttypegb:企业（机构）类型
- 53个值


cat_cols = ['oplocdistrict', 'industryphy', 'industryco', 'enttype', 'enttypeitem', 'state', 'orgid', 'jobid', 'regtype', 'opform', 'venind', 'enttypeminu', 'oploc', 'enttypegb']
two_values = ['adbusign', 'townsign', 'compform', 'protype']
num_cols = ['empnum', 'parnum', 'exenum', 'regcap', 'reccap', 'forreccap', 'forregcap', 'congro']

single_cols = ['ptbusscope', 'midpreindcode']
many_cols = ['dom', 'opscope']
dt_cols = ['opfrom', 'opto']

null_to_drop = ['midpreindcode', 'ptbusscope', 'protype', 'forreccap', 'congro', 'forregcap', 'exenum', 'parnum']

auc_to_drop = ['adbusign', 'regtype']