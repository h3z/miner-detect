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
from autoxgb import AutoXGB
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import itertools
import sys
from imblearn.over_sampling import SMOTE, ADASYN

sys.path.append("../src")
warnings.simplefilter(action="ignore", category=FutureWarning)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ds = UserJoin()


# 所有节假日合在一起算的cv
day_feature_cv_col = [
    "kwh_holiday_cv",
    "kwh_cal_holiday_cv",
    "kwh_pap_r2_holiday_cv",
    "kwh_pap_r3_holiday_cv",
    "kwh_pap_r4_holiday_cv",
    "kwh_workday_cv",
    "kwh_cal_workday_cv",
    "kwh_pap_r2_workday_cv",
    "kwh_pap_r3_workday_cv",
    "kwh_pap_r4_workday_cv",
    "kwh_cv",
    "kwh_cal_cv",
    "kwh_pap_r2_cv",
    "kwh_pap_r3_cv",
    "kwh_pap_r4_cv",
]
# 每个节假日各自算的各特征的 cv 后均值
hol_cv_col = [
    # "cv1_all",
    "cv2_all",
    "cv3_all",
    "cv4_all",
    "cv_all",
    # "cv1_holiday",
    "cv2_holiday",
    "cv3_holiday",
    "cv4_holiday",
    "cv_holiday",
    # "cv1_workday",
    "cv2_workday",
    "cv3_workday",
    "cv4_workday",
    "cv_workday",
]

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
    # "RUN_CAP",
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

all_col = (
    hol_diff_col
    + hol_mean_col
    + hol_std_col
    + day_col
    # + user_col
    + month_col
    + hol_cv_col
    + day_feature_cv_col
)


if __name__ == "__main__":
    from autoxgb import AutoXGB
    from imblearn.over_sampling import SMOTE, ADASYN

    dfall = ds.train

    df = dfall[dfall.isminer.isin(["normalt", "minert"])]

    x = df.loc[df.label != "test", all_col]
    y = df.loc[df.label != "test", "isminer"]
    x, y = ADASYN().fit_resample(x, y)
    df = pd.concat([x, y], axis=1)
    df.reset_index().to_csv("auto_xgb.csv", index=False)

    dftest = dfall[dfall.isminer.isin(["normal", "miner"])]
    dftest["isminer"] = dftest.isminer.map({"normal": "normalt", "miner": "minert"})
    dftest.reset_index().to_csv("auto_xgb_test.csv", index=False)
    # required parameters:
    train_filename = "auto_xgb.csv"
    output = "output2"

    # optional parameters
    test_filename = "auto_xgb_test.csv"
    task = "classification"
    idx = "id"
    targets = ["isminer"]
    features = all_col
    categorical_features = [
        # "ELEC_TYPE_NAME",
        # "VOLT_NAME",
    ]
    use_gpu = False
    num_folds = 5
    seed = 23
    num_trials = 5
    # time_limit = 360
    fast = False

    # Now its time to train the model!
    axgb = AutoXGB(
        train_filename=train_filename,
        output=output,
        test_filename=test_filename,
        task=task,
        idx=idx,
        targets=targets,
        features=features,
        categorical_features=categorical_features,
        use_gpu=use_gpu,
        num_folds=num_folds,
        seed=seed,
        num_trials=num_trials,
        # time_limit=time_limit,
        fast=fast,
    )
    axgb.train()
