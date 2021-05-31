import os
import sys
print(sys.path)
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score
"""
===========================================================
# 以上为导入依赖包, 所需的依赖包请在requirements.txt中列明 
-----------------------------------------------------------
# 以下为选手自定义文件和代码
===========================================================
"""

"""
===========================================================
# 选手自定义文件和代码到此为止
-----------------------------------------------------------
# 以下为main()入口函数，请谨慎修改
# main()函数必须包含to_pred_dir,result_save_path两个参数,其他参数
# 不做要求, 由选手自定义
===========================================================
"""


def main(to_pred_dir, result_save_path):
    """
        to_pred_path: 需要预测的文件夹路径
        to_save_path: 预测结果文件保存路径
    """
    iot_b_path = os.path.join(to_pred_dir, "iot_a.csv")
    payment_b_path = os.path.join(to_pred_dir, "payment_a.csv")
    orders_b_path = os.path.join(to_pred_dir, "orders_a.csv")
    """
        本示例代码省略模型过程,且假设预测结果全为1
        选手代码需要自己构建模型,并通过模型预测结果
    """
    # iot_a_path = os.path.join(sys.path[0], "iot_b.csv")
    # payment_a_path = os.path.join(sys.path[0], "payment_b.csv")
    # orders_a_path = os.path.join(sys.path[0], "orders_b.csv")

    payment_a = pd.read_csv(payment_b_path, index_col=None)
    orders_a = pd.read_csv(orders_b_path, index_col=None)
    iot_a = pd.read_csv(iot_b_path, index_col=None)

    payment_a = payment_a.loc[payment_a['Y'].notnull(), :]
    payment_a['Y'] = payment_a['Y'].map(int)
    payment_a['is_test'] = 0

    # payment_b = pd.read_csv(payment_b_path, index_col=None)
    # orders_b = pd.read_csv(orders_b_path, index_col=None)
    # iot_b = pd.read_csv(iot_b_path, index_col=None)
    # payment_b['is_test'] = 1

    payment = payment_a.copy()
    orders = orders_a.copy()
    iot = iot_a.copy()
    iot.drop(['latitude', 'longitude'], axis=1, inplace=True)
    iot['SSMONTH'] = iot['reporttime'].map(lambda x: int(x[:4]) * 100 + int(x[4:6]))
    iot['reporttime_min'] = iot.groupby(['device_code', 'SSMONTH'])['reporttime'].transform('min')
    iot['reporttime_max'] = iot.groupby(['device_code', 'SSMONTH'])['reporttime'].transform('max')
    iot_ym = iot.drop_duplicates(subset=["device_code", "SSMONTH", "reporttime_min", "reporttime_max"], keep="first")
    iot_ym.drop('reporttime', axis=1, inplace=True)
    tmp = iot.groupby(['device_code', 'SSMONTH'], as_index=False)['work_sum_time'].agg({
        'work_sum_time_nunique': 'nunique',
        'work_sum_time_cnt': 'count',
        'work_sum_time_mean': 'mean',
        'work_sum_time_sum': 'sum',
        'work_sum_time_min': 'min',
        'work_sum_time_max': 'max',
        'work_sum_time_std': 'std'
    })
    iot_ym = iot_ym.merge(tmp, on=['device_code', 'SSMONTH'], how='left')

    df1 = pd.merge(payment, orders, on=['device_code', 'customer_id'], how='left')
    df = pd.merge(df1, iot_ym, on=['device_code', 'SSMONTH'], how='left')
    print('数据合并完成!')

    df['reporttime_min-posting_date'] = (
            pd.to_datetime(df['reporttime_min']) - pd.to_datetime(df['posting_date'])).apply(lambda x: x.days)
    df['reporttime_max-posting_date'] = (
            pd.to_datetime(df['reporttime_max']) - pd.to_datetime(df['posting_date'])).apply(lambda x: x.days)
    # df['reporttime_max-reporttime_min'] = df.apply(lambda x: (pd.to_datetime(x['reporttime_max']) - pd.to_datetime(x['reporttime_min'])).days)
    df['reporttime_max-reporttime_min'] = (
            pd.to_datetime(df['reporttime_max']) - pd.to_datetime(df['reporttime_min'])).apply(lambda x: x.days)

    df.drop(['reporttime_min', 'posting_date', 'reporttime_max'], axis=1, inplace=True)
    df['DLSBH'] = df['DLSBH'].map(lambda x: int(x[-2:]))
    df['QC/RZQS'] = df['QC'] / df['RZQS']
    df['RZQS-QC'] = df['RZQS'] - df['QC']
    tmp = df.groupby(['customer_id', 'device_code'], as_index=False)['notified'].agg({
        'notified_nunique': 'nunique',
        'notified_cnt': 'count',
        'notified_mean': 'mean',
        'notified_sum': 'sum',
        'notified_min': 'min',
        'notified_max': 'max',
        'notified_std': 'std'
    })
    df = df.merge(tmp, on=['customer_id', 'device_code'], how='left')
    # df[''] = df['notified_mean'] / df['work_sum_time_mean']

    tmp = df.groupby('device_code', as_index=False)['notified'].agg({
        'device_code_notified_nunique': 'nunique',
        'device_code_notified_cnt': 'count',
        'device_code_notified_mean': 'mean',
        'device_code_notified_sum': 'sum',
        'device_code_notified_min': 'min',
        'device_code_notified_max': 'max',
        'device_code_notified_std': 'std'
    })
    df = df.merge(tmp, on='device_code', how='left')

    tmp = df.groupby('customer_id', as_index=False)['notified'].agg({
        'customer_id_notified_nunique': 'nunique',
        'customer_id_notified_cnt': 'count',
        'customer_id_notified_mean': 'mean',
        'customer_id_notified_sum': 'sum',
        'customer_id_notified_min': 'min',
        'customer_id_notified_max': 'max',
        'customer_id_notified_std': 'std'
    })
    df = df.merge(tmp, on='customer_id', how='left')

    train = df.loc[df['is_test'] == 0, :]
    test = df.loc[df['is_test'] == 1, :]
    cols = [col for col in train.columns if col not in ['device_code', 'customer_id', 'overdue']]
    train_tr = train.loc[train['SSMONTH'] != 201904, :]
    train_va = train.loc[train['SSMONTH'] == 201904, :]
    X_train = train_tr[cols]
    y_train = train_tr['Y']
    X_valid = train_va[cols]
    y_valid = train_va['Y']
    X_test = test[cols]

    cat_cols = ['DLSBH', 'SSMONTH', 'RZQS']

    dtrain = lgb.Dataset(X_train, y_train)
    dvalid = lgb.Dataset(X_valid, y_valid, reference=dtrain)

    params = {
        'objective': 'binary',
        'boosting': 'gbdt',
        # 'metric': 'auc',
        'metric': 'None',  # 用自定义评估函数是将metric设置为'None'
        'learning_rate': 0.1,
        'num_leaves': 31,
        'lambda_l1': 0,
        'lambda_l2': 1,
        'num_threads': 23,
        'min_data_in_leaf': 20,
        'first_metric_only': True,
        'is_unbalance': True,
        'max_depth': -1,
        'seed': 2021
    }

    def lgb_f1_score(y_hat, data):
        y_true = data.get_label()
        y_hat = np.where(y_hat >= 0.5, 1, 0)
        return 'f1', f1_score(y_true, y_hat), True

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dvalid, dtrain],
        num_boost_round=1000000,
        early_stopping_rounds=200,
        verbose_eval=300,
        feval=lgb_f1_score,
        categorical_feature=cat_cols
    )

    # cwd = sys.argv[0]
    # model = lgb.Booster(model_file=os.path.join(cwd[:-6], f'lgb.txt'))
    y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    print('predict Done!')

    y_pred = np.where(y_pred >= 0.5, 1, 0)

    __result = train_va.loc[:, ["SSMONTH", "device_code", "customer_id"]]
    __result["Y"] = y_pred
    # __result.sort_values(by="SSMONTH",ascending=False,inplace=True)
    result = __result[__result["SSMONTH"] == 201904]
    result.drop_duplicates(subset=["SSMONTH", "device_code", "customer_id", "Y"], keep="first", inplace=True)

    result.to_csv(result_save_path, index=None)


"""
===========================================================
# main()到此为止
-----------------------------------------------------------
# 以下代码不得修改, 若修改以下代码将会导致无法计算得分。
===========================================================
"""

# os.path.join(sys.argv[0][:-6], f'model_{i}.txt')


if __name__ == "__main__":
    to_pred_dir = sys.argv[1]  # 所需预测的文件夹路径
    result_save_path = sys.argv[2]  # 预测结果保存文件路径
    main(to_pred_dir, result_save_path)
