import warnings
import xgboost as xgb
import pandas as pd
from features.UserJoin import UserJoin
from features.UserJoin import submit, diff, plt_month, plt_day, load_ids, info, infot
import config as C
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import seaborn as sns
from xgboost import plot_tree
import random
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import itertools
import sys

sys.path.append("../src")
warnings.simplefilter(action="ignore", category=FutureWarning)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ds = UserJoin()

user = ds.train[C.user_features]

# 节假日平均和非节假日平均的差值。
hol_diff_col = [
    "kwh_holiday_diff",
    "kwh_cal_holiday_diff",
    "kwh_pap_r2_holiday_diff",
    "kwh_pap_r3_holiday_diff",
    "kwh_pap_r4_holiday_diff",
    "pr2_holiday_diff",
    "pr3_holiday_diff",
    "pr4_holiday_diff",
    "2_3_holiday_diff",
    "2_4_holiday_diff",
    "3_4_holiday_diff",
    "daycv_holiday_diff",
]

# 节假日均值 和 非节假日的均值
hol_mean_col = [
    "kwh_holiday_mean_1",
    "kwh_cal_holiday_mean_1",
    "kwh_pap_r2_holiday_mean_1",
    "kwh_pap_r3_holiday_mean_1",
    "kwh_pap_r4_holiday_mean_1",
    "pr2_holiday_mean_1",
    "pr3_holiday_mean_1",
    "pr4_holiday_mean_1",
    "2_3_holiday_mean_1",
    "2_4_holiday_mean_1",
    "3_4_holiday_mean_1",
    "daycv_holiday_mean_1",
    "kwh_holiday_mean_0",
    "kwh_cal_holiday_mean_0",
    "kwh_pap_r2_holiday_mean_0",
    "kwh_pap_r3_holiday_mean_0",
    "kwh_pap_r4_holiday_mean_0",
    "pr2_holiday_mean_0",
    "pr3_holiday_mean_0",
    "pr4_holiday_mean_0",
    "2_3_holiday_mean_0",
    "2_4_holiday_mean_0",
    "3_4_holiday_mean_0",
    "daycv_holiday_mean_0",
]

# 节假日方差 和 非节假日的方差
hol_std_col = [
    "kwh_holiday_std_1",
    "kwh_cal_holiday_std_1",
    "kwh_pap_r2_holiday_std_1",
    "kwh_pap_r3_holiday_std_1",
    "kwh_pap_r4_holiday_std_1",
    "pr2_holiday_std_1",
    "pr3_holiday_std_1",
    "pr4_holiday_std_1",
    "2_3_holiday_std_1",
    "2_4_holiday_std_1",
    "3_4_holiday_std_1",
    "daycv_holiday_std_1",
    "kwh_holiday_std_0",
    "kwh_cal_holiday_std_0",
    "kwh_pap_r2_holiday_std_0",
    "kwh_pap_r3_holiday_std_0",
    "kwh_pap_r4_holiday_std_0",
    "pr2_holiday_std_0",
    "pr3_holiday_std_0",
    "pr4_holiday_std_0",
    "2_3_holiday_std_0",
    "2_4_holiday_std_0",
    "3_4_holiday_std_0",
    "daycv_holiday_std_0",
]

# 日基础数据
day_col = [
    "kwh",
    "kwh_cal",
    "kwh_pap_r2",
    "kwh_pap_r3",
    "kwh_pap_r4",
    "pr2",
    "pr3",
    "pr4",
    "2_3",
    "2_4",
    "3_4",
    "daycv",
]

user_col = [
    "ELEC_TYPE_NAME",
    "VOLT_NAME",
    "RUN_CAP",
    # 'ELEC_TYPE_NAME_CODE',
    # 'VOLT_NAME_CODE',
    # 'label'
]

month_col = [
    "pq_f",
    "pq_g",
    "pq_p",
    "pq_z",
    "pp",
    "pf",
    "pg",
    "p_f",
    "p_g",
    "f_g",
    "monthcv",
]

all_col = hol_diff_col + hol_mean_col + hol_std_col + day_col + user_col + month_col


def exp(features, df, params):
    x, x_val = (
        df.loc[df.label != "test", features],
        df.loc[df.label == "test", features],
    )

    if "ELEC_TYPE_NAME" in features:
        x = pd.concat([x, pd.get_dummies(x.ELEC_TYPE_NAME)], axis=1).drop(
            columns=["ELEC_TYPE_NAME"]
        )
        x_val = pd.concat([x_val, pd.get_dummies(x_val.ELEC_TYPE_NAME)], axis=1).drop(
            columns=["ELEC_TYPE_NAME"]
        )

    if "VOLT_NAME" in features:
        x = pd.concat([x, pd.get_dummies(x.VOLT_NAME)], axis=1).drop(
            columns=["VOLT_NAME"]
        )
        x_val = pd.concat([x_val, pd.get_dummies(x_val.VOLT_NAME)], axis=1).drop(
            columns=["VOLT_NAME"]
        )

    y = df.loc[df.label != "test", "label"].astype("int")
    model = xgb.XGBClassifier(verbosity=0, **params)
    model.fit(x, y)

    y_val = model.predict(x_val)
    pred_val = pd.DataFrame({"id": x_val.index.values, "pred": y_val})
    return pred_val


def today_exp(params):
    t = ds.train2

    pred_val = exp(all_col, t, params)

    ids = set(pred_val.query("pred == 1").id.unique())
    return [
        0,
        len(ids & C.true_ids),
        len(ids & C.false_ids),
        len(ids - C.true_ids - C.false_ids),
        len(ids),
        str(ids),
    ]


if __name__ == "__main__":
    params = {
        "max_depth": [3, 5, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "n_estimators": [100, 200, 300, 500],
        "colsample_bytree": [0.3, 0.4, 0.6, 0.8, 1],
        "subsample": [0.4, 0.8, 1],
    }

    dfs = []
    for max_depth, learning_rate, n_estimators, colsample_bytree, subsample in tqdm(
        list(
            itertools.product(
                params["max_depth"],
                params["learning_rate"],
                params["n_estimators"],
                params["colsample_bytree"],
                params["subsample"],
            )
        )
    ):
        p = {
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "colsample_bytree": colsample_bytree,
            "subsample": subsample,
        }
        res = today_exp(p)
        p["total"] = res[4]
        p["right_miner"] = res[1]
        p["wrong_miner"] = res[2]
        p["unknown"] = res[3]
        p["ids"] = res[5]

        dfs.append(pd.DataFrame(p, index=[0]))

        df = pd.concat(dfs)
        df.to_csv("test_on_user.csv", index=False)
