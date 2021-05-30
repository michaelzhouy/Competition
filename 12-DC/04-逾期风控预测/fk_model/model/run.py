import os
import sys
print(sys.path)
import numpy as np
import pandas as pd
import lightgbm as lgb
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
def main(to_pred_dir,result_save_path):
    """
        to_pred_path: 需要预测的文件夹路径
        to_save_path: 预测结果文件保存路径
    """
    iot_path = os.path.join(to_pred_dir, "iot_b.csv")
    payment_path = os.path.join(to_pred_dir, "payment_b.csv")
    orders_path = os.path.join(to_pred_dir, "orders_b.csv")
    """
        本示例代码省略模型过程,且假设预测结果全为1
        选手代码需要自己构建模型,并通过模型预测结果
    """

    # to_pred_file_list = [os.path.join(to_pred_dir,f) for f in os.listdir(to_pred_dir)]

    payment_a = pd.read_csv(payment_path, index_col=None)
    orders_a = pd.read_csv(orders_path, index_col=None)

    payment_a = pd.merge(payment_a, orders_a, on=['device_code', 'customer_id'], how='left')
    payment_a['today'] = '2021-05-30'

    payment_a['posting_date_diff'] = (
                pd.to_datetime(payment_a['today']) - pd.to_datetime(payment_a['posting_date'])).apply(lambda x: x.days)
    payment_a['SSMONTH_str'] = payment_a['SSMONTH'].map(lambda x: str(x)[:4] + '-' + str(x)[4:6] + '-' + '01')
    payment_a['SSMONTH-posting_date'] = (pd.to_datetime(payment_a['SSMONTH_str']) - pd.to_datetime(payment_a['posting_date'])).apply(lambda x: x.days)
    payment_a.drop(['today', 'posting_date', 'SSMONTH_str'], axis=1, inplace=True)
    payment_a['DLSBH'] = payment_a['DLSBH'].map(lambda x: int(x[-2:]))
    payment_a['QC/RZQS'] = payment_a['QC'] / payment_a['RZQS']
    payment_a['RZQS-QC'] = payment_a['RZQS'] - payment_a['QC']

    X = payment_a.drop(['device_code', 'customer_id', 'overdue', 'Y'], axis=1)

    cwd = sys.argv[0]
    print('cwd-----------')
    print(cwd)
    model = lgb.Booster(model_file=os.path.join(cwd[:-6], f'lgb.txt'))
    y_pred = model.predict(X, num_iteration=model.best_iteration)
    print('predict Done!')

    y_pred = np.where(y_pred >= 0.5, 1, 0)

    __result = payment_a.loc[:, ["SSMONTH", "device_code", "customer_id"]]
    __result["Y"] = y_pred
    #__result.sort_values(by="SSMONTH",ascending=False,inplace=True)
    result = __result[__result["SSMONTH"]==201904]
    result.drop_duplicates(subset=["SSMONTH","device_code","customer_id","Y"],keep="first",inplace=True)
    
    result.to_csv(result_save_path,index=None)
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
    main(to_pred_dir,result_save_path)
