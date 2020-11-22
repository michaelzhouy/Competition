# -*- coding: utf-8 -*-
# @Time     : 2020/11/22 20:15
# @Author   : Michael_Zhouy

str_cols = ['srcAddress', 'destAddress', 'destGeoCountry',
   'destGeoRegion', 'destGeoCity', 'catOutcome', 'destHostName',
   'requestMethod', 'httpVersion', 'requestUrlQuery', 'requestUrl',
   'httpReferer', 'accessAgent', 'requestBody', 'requestHeader',
   'responseHeader', 'requestContentType', 'responseContentType']
res={}
for i in str_cols:
    for j in str_cols:
        if j==i:
            continue
        tr = train[i].groupby(train[j]).agg(['count','nunique'])
        train['{}_gp_{}_count'.format(i,j)] = train[j].map(tr['count'])
        train['{}_gp_{}_nunique'.format(i,j)] = train[j].map(tr['nunique'])
        res['{}_gp_{}'.format(i,j)]=tr
        train['{}_gp_{}_nunique_rate'.format(i,j)] = train['{}_gp_{}_nunique'.format(i,j)]/train['{}_gp_{}_count'.format(i,j)]


str_cols = ['srcAddress', 'destAddress', 'destGeoCountry',
   'destGeoRegion', 'destGeoCity', 'catOutcome', 'destHostName',
   'requestMethod', 'httpVersion', 'requestUrlQuery', 'requestUrl',
   'httpReferer', 'accessAgent', 'requestBody', 'requestHeader',
   'responseHeader', 'requestContentType', 'responseContentType']
for i in str_cols:
    for j in str_cols:
        if j==i:
            continue
        tr = res['{}_gp_{}'.format(i,j)]
        test_1['{}_gp_{}_count'.format(i,j)] = test_1[j].map(tr['count'])
        test_1['{}_gp_{}_nunique'.format(i,j)] = test_1[j].map(tr['nunique'])
        test_1['{}_gp_{}_nunique_rate'.format(i,j)] = test_1['{}_gp_{}_nunique'.format(i,j)]/test_1['{}_gp_{}_count'.format(i,j)]


joblib.dump(res,'./res.pkl')
joblib.load('./res.pkl)