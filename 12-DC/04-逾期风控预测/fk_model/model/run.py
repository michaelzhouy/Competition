import os
import sys
print(sys.path)
import numpy as np
import pandas as pd
import lightgbm as lgb
import calendar
import datetime
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
    iot_b_path = os.path.join(to_pred_dir, "iot_b.csv")
    payment_b_path = os.path.join(to_pred_dir, "payment_b.csv")
    orders_b_path = os.path.join(to_pred_dir, "orders_b.csv")
    train_path = sys.path[0] + '/'

    payment_a = pd.read_csv(payment_b_path, index_col=None)
    orders_a = pd.read_csv(orders_b_path, index_col=None)
    iot_a = pd.read_csv(iot_b_path, index_col=None)

    payment_a.sort_values(['customer_id', 'device_code', 'SSMONTH'], ascending=True, inplace=True)
    payment_a = pd.merge(payment_a, orders_a, on=['device_code', 'customer_id'], how='left')

    payment_a['posting_date'] = payment_a['posting_date'].map(
        lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())
    payment_a['posting_date_month_end'] = payment_a['posting_date'].map(
        lambda x: datetime.datetime(x.year, x.month, calendar.monthrange(x.year, x.month)[1]).date())
    payment_a['month_end_diff'] = (payment_a['posting_date_month_end'] - payment_a['posting_date']).map(
        lambda x: x.days)
    payment_a['posting_date_weekday'] = payment_a['posting_date'].map(lambda x: x.weekday())
    # payment_a['posting_date_weekend'] = payment_a['posting_date_weekday'].map(lambda x: 1 if x in [5, 6] else 0)

    payment_a.drop(['posting_date', 'posting_date_month_end'], axis=1, inplace=True)

    # payment_a['year'] = payment_a['SSMONTH'].map(lambda x: int(str(x)[:4]))
    payment_a['month'] = payment_a['SSMONTH'].map(lambda x: int(str(x)[4:6]))
    payment_a['month_chunjie'] = payment_a['month'].map(lambda x: 1 if x in [1, 11, 12] else 0)

    payment_a['DLSBH'] = payment_a['DLSBH'].map(lambda x: int(x[-2:]))

    payment_a['RZQS'] = payment_a['RZQS'].map(int)
    payment_a['QC'] = payment_a['QC'].map(int)

    payment_a['RZQS-QC'] = payment_a['RZQS'] - payment_a['QC']  # 剩余期次数
    payment_a['QC/RZQS'] = payment_a['QC'] / payment_a['RZQS']  # 当前期次 / 融资期数

    # payment_a['notified_shift_1'] = payment_a.groupby(['customer_id', 'device_code'])['notified'].shift(1)
    # payment_a['notified_shift_2'] = payment_a.groupby(['customer_id', 'device_code'])['notified'].shift(2)
    # payment_a['notified_2'] = payment_a['notified'] + payment_a['notified_shift_1']
    # payment_a['notified_shift_-1'] = payment_a.groupby(['customer_id', 'device_code'])['notified'].shift(-1)
    # payment_a['notified_shift_-2'] = payment_a.groupby(['customer_id', 'device_code'])['notified'].shift(-2)
    # fill = {'notified_shift_1': 1, 'notified_shift_2': 1}
    # payment_a.fillna(fill, inplace=True)

    # payment_a['customer_id_notified_times'] = payment_a.groupby('customer_id')['notified'].transform('mean')
    # payment_a['device_code_notified_times'] = payment_a.groupby('device_code')['notified'].transform('mean')
    # payment_a['device_code_customer_id_notified_times'] = payment_a.groupby(['device_code', 'customer_id'])[
    #     'notified'].transform('mean')
    # payment_a['DLSBH_customer_id_notified_times'] = payment_a.groupby(['DLSBH', 'customer_id'])['notified'].transform(
    #     'mean')
    # payment_a['DLSBH_device_code_notified_times'] = payment_a.groupby(['DLSBH', 'device_code'])['notified'].transform(
    #     'mean')

    X_test = payment_a.drop(['device_code', 'customer_id', 'overdue', 'Y'], axis=1)

    train_path = sys.path[0] + '/'
    model = lgb.Booster(model_file=train_path + 'lgb.txt')
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    print('predict Done!')

    y_pred = np.where(y_pred >= 0.49, 1, 0)

    __result = payment_a.loc[:, ["SSMONTH", "device_code", "customer_id"]]
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


if __name__ == "__main__":
    to_pred_dir = sys.argv[1]  # 所需预测的文件夹路径
    result_save_path = sys.argv[2]  # 预测结果保存文件路径
    main(to_pred_dir, result_save_path)
