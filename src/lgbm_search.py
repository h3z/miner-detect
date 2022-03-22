import itertools
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from pathlib import Path
import random
from xgboost import plot_tree
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import config as C
from features.UserJoin import submit, diff, plt_month, plt_day, load_ids, info, infot
from features.UserJoin import UserJoin
import pandas as pd
import xgboost as xgb
import warnings
import lightgbm as lgbm

warnings.simplefilter(action="ignore", category=FutureWarning)

from sklearn.metrics import f1_score


def score(right, wrong):
    right = int(right)
    wrong = int(wrong)
    true_num = 37
    y_true = np.zeros(15379)
    y_pred = np.zeros(15379)
    y_true[:true_num] = 1
    y_pred[:right] = 1
    if wrong:
        y_pred[-wrong:] = 1
    score = f1_score(y_true, y_pred, average="macro")
    return score


ds = UserJoin()


def preprocess(df, cols):
    if "ELEC_TYPE_NAME" in cols:
        df = pd.concat([df, pd.get_dummies(df.ELEC_TYPE_NAME)], axis=1).drop(
            columns=["ELEC_TYPE_NAME"]
        )
    if "VOLT_NAME" in cols:
        df = pd.concat([df, pd.get_dummies(df.VOLT_NAME)], axis=1).drop(
            columns=["VOLT_NAME"]
        )

    features = [i for i in cols if i not in ["ELEC_TYPE_NAME", "VOLT_NAME"]]
    x, x_val = (
        df.loc[df.label != "test", features],
        df.loc[df.label == "test", features],
    )
    y = df.loc[df.label != "test", "label"].astype("int")

    return x, x_val, y


def exp(features, df, params):
    x, x_val, y = preprocess(df, features)

    # model = xgb.XGBClassifier(verbosity=0, **params)
    model = lgbm.LGBMClassifier(max_depth=1, **params)
    model.fit(x, y)
    pred_y = model.predict(x)
    # print((pred_y != y).sum())

    y_val = model.predict(x_val)
    pred_val = (
        pd.DataFrame({"id": x_val.index.values, "pred": y_val}).groupby("id").sum()
    )
    pred_y = pd.DataFrame(
        {
            "id": x.index.values,
            "ym": df.loc[df.label != "test"].ym.values,
            "pred": pred_y,
            "label": y.values,
        }
    )
    return pred_y, pred_val


def today_exp(params):

    t = ds.month
    cols = C.month_features
    pred_y, pred_val = exp(cols, t, params)

    res = []
    for i in range(22):
        ids = set(pred_val[pred_val.pred > i].index.values)
        res.append(
            [
                i,
                len(ids & C.true_ids),
                len(ids - C.true_ids),
                len(ids),
                str(ids),
            ]
        )
    return res


if __name__ == "__main__":

    # * n_estimator 100
    # * lr 0.05
    # * reg_alpha, reg_labmda .1
    params = {
        # "learning_rate": [0.05],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        # "n_estimators": [100, 200, 300, 500],
        "n_estimators": [75, 85],
        "colsample_bytree": [0.3, 0.4, 0.6, 0.8, 1],
        "reg_alpha": [1e-3, 1e-2, 1e-1, 0, 0.5, 1],
        "reg_lambda": [1e-3, 1e-2, 1e-1, 0, 0.5, 1],
        # "reg_alpha": [0.1],
        # "reg_lambda": [0.1],
        # "reg_alpha": [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        # "reg_lambda": [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    }

    dfs = []
    for learning_rate, n_estimators, colsample_bytree, reg_alpha, reg_lambda in tqdm(
        list(
            itertools.product(
                params["learning_rate"],
                params["n_estimators"],
                params["colsample_bytree"],
                params["reg_alpha"],
                params["reg_lambda"],
            )
        )
    ):
        p = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
        }
        res = today_exp(p)
        for i in res:
            p["gt"] = i[0]
            p["right"] = i[1]
            p["wrong"] = i[2]
            p["total"] = i[3]
            p["f1score"] = score(i[1], i[2])
            p["ids"] = i[4]

            dfs.append(pd.DataFrame(p, index=[0]))

        df = pd.concat(dfs)
        df.to_csv("lgbm_search_75_85.csv", index=False)
