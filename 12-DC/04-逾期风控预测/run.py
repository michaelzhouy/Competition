import os
import sys

import numpy as np

print(sys.path)
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

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
    iot_path = os.path.join(to_pred_dir, "iot_b.csv")
    payment_path = os.path.join(to_pred_dir, "payment_b.csv")
    orders_path = os.path.join(to_pred_dir, "orders_b.csv")
    """
        本示例代码省略模型过程,且假设预测结果全为1
        选手代码需要自己构建模型，并通过模型预测结果
    """
    train_path = sys.path[0] + '/'
    payment_a = pd.read_csv(train_path + 'payment_a.csv', index_col=None)
    orders_a = pd.read_csv(train_path + 'orders_a.csv')
    iot_a = pd.read_csv(train_path + 'iot_a.csv')

    payment_a = payment_a[payment_a['overdue'] == 0].reset_index(drop=True)
    payment_b = pd.read_csv(payment_path, index_col=None)
    payment_a['flag'] = 'train'
    payment_b['flag'] = 'test'
    payment_a['Y'] = payment_a['Y'].astype(float)
    payment = pd.concat([payment_a, payment_b], axis=0, copy=False)

    # for col in ['device_code', 'customer_id', 'DLSBH']:
    #     payment[col] = payment[col].astype('category')
    #     payment[col] = payment[col].cat.codes

    # Label Encoder
    for i in ['device_code', 'customer_id']:
        payment[i] = payment[i].map(dict(zip(payment[i].unique(), range(payment[i].nunique()))))

    payment['DLSBH'] = payment['DLSBH'].map(lambda x: int(x[-2:]))

    payment['QC/RZQS'] = payment['QC'] / (payment['RZQS'] + 0.00001)
    payment['RZQS-QC'] = payment['RZQS'] - payment['QC']

    cat_cols = ['DLSBH', 'SSMONTH', 'RZQS', 'QC']
    feature = [col for col in payment.columns if col not in ['device_code', 'customer_id', 'Y', 'flag', 'overdue']]

    train_total_data = payment[payment['flag'] == 'train']
    x_train = train_total_data[train_total_data['SSMONTH'] != 201904][feature]
    y_train = train_total_data[train_total_data['SSMONTH'] != 201904]['Y']
    x_valid = train_total_data[train_total_data['SSMONTH'] == 201904][feature]
    y_valid = train_total_data[train_total_data['SSMONTH'] == 201904]['Y']
    xx_score = train_total_data[train_total_data['SSMONTH'] == 201904][['Y']]
    test_data = payment[payment['flag'] == 'test'][feature]

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'max_depth': -1,
        "learning_rate": 0.1,
        'n_jobs': -1,
        'seed': 42,
    }

    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train)
    gbm = lgb.train(params, lgb_train, num_boost_round=10000, valid_sets=[lgb_train, lgb_eval],
                    verbose_eval=50,
                    early_stopping_rounds=50, categorical_feature=cat_cols)
    vaild_preds = gbm.predict(x_valid, num_iteration=gbm.best_iteration)
    test_preds = gbm.predict(test_data, num_iteration=gbm.best_iteration)
    test_preds = np.where(test_preds >= 0.5, 1, 0)

    xx_score_one = xx_score[xx_score['Y'] == 1].shape[0]

    xx_score['pred'] = vaild_preds
    xx_score['rank'] = xx_score['pred'].rank(ascending=False)
    xx_score['Y_1'] = 0
    xx_score.loc[xx_score['rank'] <= int(xx_score_one), 'Y_1'] = 1
    best = f1_score(xx_score['Y'], xx_score['Y_1'])
    print(best)

    __result = payment_b.loc[:, ["SSMONTH", "device_code", "customer_id", "Y"]]

    __result['pred'] = test_preds
    __result['rank'] = __result['pred'].rank(ascending=False)
    __result['Y'] = test_preds
    # __result.loc[__result['rank'] <= int(__result.shape[0] * 0.2), 'Y'] = 1
    result = __result[__result["SSMONTH"] == 201904]
    result.drop_duplicates(subset=["SSMONTH", "device_code", "customer_id"], keep="first", inplace=True)
    result[["SSMONTH", "device_code", "customer_id", "Y"]].to_csv(result_save_path, index=None)


"""
===========================================================
# main()到此为止
-----------------------------------------------------------
# 以下代码不得修改, 若修改以下代码将会导致无法计算得分。
===========================================================
"""

if __name__ == "__main__":
    # to_pred_dir = '../test/'  # 所需预测的文件夹路径
    to_pred_dir = sys.argv[1]  # 所需预测的文件夹路径
    # result_save_path = '../test/'  # 预测结果保存文件路径
    result_save_path = sys.argv[2]  # 预测结果保存文件路径
    main(to_pred_dir, result_save_path)