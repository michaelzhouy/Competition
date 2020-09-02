# -*- coding: utf-8 -*-
# @Time     : 2020/8/31 20:47
# @Author   : Michael_Zhouy

# 并行计算
def process(gps):
    def start_end(df):
        '''
        起点经度，起点纬度，起点时间，终点时间
        '''
        print('正在处理records...')
        df['gps_records'] = df['gps_records'].map(lambda x: x.replace('[', '').replace(']', '').split(','))

    start_end(gps)
    gc.collect()
    print(gps.shape)

    def speed_grid(t_trace, res):
        # 遍历轨迹中的每个点
        for i in range(len(t_trace)):
            # 点信息
            info = t_trace[i].strip().split(" ")
            t_x = float(info[0])
            t_y = float(info[1])
            t_direct = float(info[3])
            # 记录时间
            t_time = datetime.fromtimestamp(int(info[4])).strftime('%Y-%m-%d %H:%M')[:15] + '0:00'
            # 遍历网格
            if t_time in res.index:
                for x_i in range(bin_num):
                    if t_x >= x_list[x_i] and t_x < x_list[x_i + 1]:
                        for y_i in range(bin_num):
                            if t_y >= y_list[y_i] and t_y < y_list[y_i + 1]:
                                res.loc[t_time]['grid_x{}y{}'.format(x_i, y_i)] += 1
                                break
                        break
        return res

    print('提取gps信息...')

    def tmp_func(df1):
        res = pd.DataFrame(np.zeros((len(date_range), len(cols_))),
                           columns=cols_,
                           index=date_range)
        res.index = [str(x) for x in res.index]
        df1['gps_records'].apply(lambda x: speed_grid(x, res))
        return res

        # 并行加速统计

    df_grouped = gps.groupby(gps.rand)
    result = Parallel(n_jobs=20)(delayed(tmp_func)(group) for name, group in tqdm(df_grouped))

    resf = pd.DataFrame(np.zeros((len(date_range), len(cols_))),
                        columns=cols_,
                        index=date_range)
    resf.index = [str(x) for x in resf.index]
    print('合并gps信息')
    for res_i in result:
        resf += res_i
    print('提取完成.../n')
    return resf


# 分块处理,每次处理5万行
res_f = pd.DataFrame(np.zeros((len(date_range), len(cols_))),
                     columns=cols_,
                     index=date_range)
res_f.index = [str(x) for x in res_f.index]

for gps in tqdm(pd.read_csv(data_path + t_file, chunksize=50000, names=['id_user', 'id_order', 'gps_records'])):
    gps['rand'] = np.random.randint(20, size=len(gps))
    print(gps.shape)
    res_t = process(gps)
    res_f += res_t
    gc.collect()
res_f.to_hdf(data_path + t_to_file, 'df')
gc.collect()