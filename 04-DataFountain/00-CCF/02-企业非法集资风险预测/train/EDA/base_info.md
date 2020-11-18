1. id:企业唯一标识,
2. oplocdistrict:行政区划代码
3. industryphy:行业类别代码
4. industryco:行业细类代码
5. dom:经营地址
6. opscope:经营范围
7. enttype:企业类型
8. enttypeitem:企业类型小类
9. opfrom:经营期限起
10. opto:经营期限止
11. state:状态
12. orgid:机构标识
13. jobid:职位标识
14. adbusign:是否广告经营
15. townsign:是否城镇
16. regtype:主题登记类型
17. empnum:从业人数
18. compform:组织形式
19. parnum:合伙人数
20. exenum:执行人数
21. opform:经营方式
22. ptbusscope:兼营范围
23. venind:风险行业
24. enttypeminu:企业类型细类
25. midpreindcode:中西部优势产业代码
26. protype:项目类型
27. oploc:经营场所
28. regcap:注册资本（金）
29. reccap:实缴资本
30. forreccap:实缴资本（外方）
31. forregcap:注册资本（外方）
32. congro:投资总额
33. enttypegb:企业（机构）类型


cat_cols = ['oplocdistrict', 'industryphy', 'industryco', 'enttype', 'enttypeitem', 'state', 'orgid', 'jobid', 'regtype', 'opform', 'venind', 'enttypeminu', 'oploc', 'enttypegb']
two_values = ['adbusign', 'townsign', 'compform', 'protype']
num_cols = ['empnum', 'parnum', 'exenum', 'regcap', 'reccap', 'forreccap', 'forregcap', 'congro']

single_cols = ['ptbusscope', 'midpreindcode']
many_cols = ['dom', 'opscope']
dt_cols = ['opfrom', 'opto']

null_to_drop = ['midpreindcode', 'ptbusscope', 'protype', 'forreccap', 'congro', 'forregcap', 'exenum', 'parnum']

auc_to_drop = ['adbusign', 'regtype']