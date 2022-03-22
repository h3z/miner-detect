from matplotlib.cbook import ls_mapper
from features.MonthData import MonthData
from features.UsersData import UsersData
from features.DayData import DayData
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import config as C
import plotly.express as px
from pathlib import Path
from sklearn.metrics import f1_score


def gen_day_hol_n(df):
    # nth holiday tag
    hol_n_dct = hol_n_dictionary(df)
    return df.rq.map(hol_n_dct)


def hol_n_dictionary(df):
    rq_lst = df.rq.sort_values().unique()
    dct = {}
    count = 0
    dct[rq_lst[0]] = count

    for i in range(1, len(rq_lst)):
        if (rq_lst[i] - rq_lst[i - 1]) != np.timedelta64(1, "D"):
            count += 1
        dct[rq_lst[i]] = count

    return dct


def f1score(ids):
    label = pd.DataFrame(
        {
            "id": list(C.true_ids) + list(C.false_ids),
            "label": [1] * len(C.true_ids) + [0] * len(C.false_ids),
        }
    ).set_index("id")
    join = label.join(
        pd.DataFrame({"id": list(ids), "pred": 1}).set_index("id")
    ).fillna(0)
    return f1_score(join.label.values, join.pred.values, average="macro")


def infot(target):
    for i in Path("submit_csv").glob("3_10_base*.csv"):
        cmp = load_ids(i.name)
        print(i.name, target - cmp, len(target & cmp))


def info(f):
    print(f)
    target = load_ids(f)
    infot(target)


def plt_day(allids, df, cols=["pr2", "pr3", "pr4"], x="rq"):
    allids = list(allids)
    dct = {}
    for i, v in enumerate(df[x].sort_values().unique()):
        dct[v] = i

    if type(allids) is str:
        t = pd.read_csv(allids)
        allids = t[t.label == 1].id.unique()

    if "id" not in df.columns:
        df["id"] = df.index

    def f(ids):
        print(ids)
        d = df[df.id.isin(ids)]
        d = d.replace({x: dct})
        if d[cols[0]].max() > 1:
            fig = px.line(
                d,
                x=x,
                y=cols,
                facet_col="id",
                facet_col_wrap=4,
                height=(int((len(ids) - 1) / 4) + 1) * 300,
            )
        else:
            fig = px.line(
                d,
                x=x,
                y=cols,
                facet_col="id",
                facet_col_wrap=4,
                height=(int((len(ids) - 1) / 4) + 1) * 300,
                range_y=(0, 1),
            )
        fig.show()

    for i in range(int(((len(allids) - 1) / 20) + 1)):
        if i * 20 + 20 > len(allids):
            f(allids[i * 20 :])
        else:
            f(allids[i * 20 : i * 20 + 20])


def plt_month(allids, df, cols=["pp", "pf", "pg"], x="ym"):
    allids = list(allids)

    if type(allids) is str:
        t = pd.read_csv(allids)
        allids = t[t.label == 1].id.unique()

    if "id" not in df.columns:
        df["id"] = df.index

    def f(ids):
        print(ids)
        d = df[df.id.isin(ids)]
        if d[cols[0]].max() > 1:
            fig = px.line(
                d,
                x=x,
                y=cols,
                facet_col="id",
                facet_col_wrap=4,
                height=(int((len(ids) - 1) / 4) + 1) * 300,
            )
        else:
            fig = px.line(
                d,
                x=x,
                y=cols,
                facet_col="id",
                facet_col_wrap=4,
                height=(int((len(ids) - 1) / 4) + 1) * 300,
                range_y=(0, 1),
            )
        fig.show()

    for i in range(int(((len(allids) - 1) / 20) + 1)):
        if i * 20 + 20 > len(allids):
            f(allids[i * 20 :])
        else:
            f(allids[i * 20 : i * 20 + 20])


def check_f(s, df, n=2):
    def norm_arr(s):
        return s[((s < (s.mean() + n * s.std())) & (s > (s.mean() - n * s.std())))]

    norm_arr(df[s]).hist(bins=60, alpha=0.4)

    s2 = df[df.index.isin(C.minerids)][s]
    s2.hist(bins=60)
    print(s2.min(), s2.max())
    plt.axvline(x=s2.min(), c="r")
    plt.axvline(x=s2.max(), c="r")
    plt.title(s)
    plt.show()


def to_cat_code(df, col):
    df[col] = df[col].astype("category")
    df[col + "_CODE"] = df[col].cat.codes
    return df


def scale(df):
    scale = 0.05
    std_max = 0.05
    return df[
        (df.pp < 0.333 + scale)
        & (df.pp > 0.333 - scale)
        & (df.pf < 0.333 + scale)
        & (df.pf > 0.333 - scale)
        & (df.pg < 0.333 + scale)
        & (df.pg > 0.333 - scale)
    ]


def to_path(f):
    if "backup" in f:
        f = f.replace("backup", "")
        return f"../submit_csv/{f}"
    return f"submit_csv/{f}"


def load_ids(f):
    d = pd.read_csv(to_path(f))
    return set(d[d.label == 1].id.values)


def diff_set(s1, s2):
    print("in s1 not s2", len(s1 - s2))
    print("in s2 not s1", len(s2 - s1))
    print("inner", len(s2 & s1))

    print(len(s1 & s2) / (len(s1) + len(s2) - len(s1 & s2)))
    return s1 & s2


def diff(f1, f2):
    print()
    dict = {
        "submit_3_3_1.csv": 33,
        "submit_3_4_1.csv": 29,
        "submit_3_4_2.csv": 20,
        "submit_3_4_3.csv": 19,
        "base.csv": 27,
        "submit_3_5_1.csv": 22,
        "submit_3_5_2.csv": 21,
        "submit_3_7_1.csv": 27,
        "submit_3_7_2.csv": 23,
        "submit_3_7_3.csv": 22,
        "submit_3_8_1.csv": 11,
        "submit_3_8_2.csv": 27,
        "submit_3_8_3.csv": 20,
        "submit_3_9_1.csv": 34,
        "submit_3_9_2.csv": 24,
    }

    d1 = pd.read_csv(to_path(f1))
    d2 = pd.read_csv(to_path(f2))

    s1 = set(d1[d1.label == 1].id.values)
    s2 = set(d2[d2.label == 1].id.values)

    if "backup" in f1:
        f1 = f1.replace("backup", "")
    if "backup" in f2:
        f2 = f2.replace("backup", "")

    if f1 in dict:
        print(f"\n{f1}", dict[f1], end="")
    if f2 in dict:
        print(f"\n{f2}", dict[f2], end="")

    print()
    print(f"{f1} 比 {f2} {len(s1 & s2) / (len(s1) + len(s2) - len(s1 & s2)):.3f}")
    print(f"多了: ", len(s1 - s2))
    print(f"少了: ", len(s2 - s1))
    print("都有: ", len(s2 & s1))

    print()
    return s1, s2


def submit(ids, f, reallysubmit=False):
    res = []
    for id in UsersData(True).df.index.values:
        res.append([id, int(id in ids)])

    df = pd.DataFrame(res, columns=["id", "label"])
    print(df.label.sum())
    if reallysubmit and f.startswith("submit"):
        df.to_csv(to_path(f), index=False)
    else:
        df.to_csv(to_path(f if f.startswith("t") else "t" + f))
    return df


class UserJoin:
    def __init__(self, with_aug=False) -> None:
        user, usert = UsersData(False, with_aug).df, UsersData(True).df
        user["label"] = user["IS_FLAG"].astype("str")
        usert["label"] = "test"
        user = pd.concat([user, usert])

        self.day = self.gen_day(user, with_aug)
        self.month = self.gen_month(user, with_aug)
        self.day_mean = self.gen_day_mean()
        self.month_mean = self.gen_month_mean()

        self.holiday_cv = self.gen_holiday_cv()

        # join to ds
        train = (
            user[["ELEC_TYPE_NAME", "VOLT_NAME", "RUN_CAP", "label"]]
            .join(self.day_mean)
            .join(self.month_mean)
            .join(self.holiday_cv.reset_index().set_index("id"))
            .dropna()
        )  # 只有两个 na，先drop 掉
        train.index.name = "id"

        # train = to_cat_code(train, 'ELEC_TYPE_NAME')
        # train = to_cat_code(train, 'VOLT_NAME')
        # train = train[(train.pq_p >= 0) &
        #               (train.pq_g >= 0) &
        #               (train.pq_z >= 0)]

        self.train = train
        # self.train = self.rule1(train)
        self.train0 = self.rule1(self.train)
        self.train1 = self.rule2(self.train0)
        self.train2 = self.rule3(self.train1)

    def rule1(self, df):
        return df[df.VOLT_NAME == "交流380V"]

    def rule2(self, df):
        return df[df.kwh > 50]

    def rule3(self, df, scale=0.06):
        std_max = 0.05
        return df[
            (df.pp < 0.333 + scale)
            & (df.pp > 0.333 - scale)
            & (df.pf < 0.333 + scale)
            & (df.pf > 0.333 - scale)
            & (df.pg < 0.333 + scale)
            & (df.pg > 0.333 - scale)
        ]

    def gen_month(self, user, with_aug):
        month = pd.concat([MonthData(False, with_aug).df, MonthData(True).df]).join(
            user
        )
        month.index.name = "id"

        # month minus
        month["p_f"] = np.abs(month.pp - month.pf)
        month["p_g"] = np.abs(month.pp - month.pg)
        month["f_g"] = np.abs(month.pf - month.pg)
        month["monthcv"] = month[["pq_g", "pq_p", "pq_f"]].std(axis=1) / (
            month[["pq_g", "pq_p", "pq_f"]].mean(axis=1) + 1e-5
        )

        month = month[month.ym != "2020-10"]

        return month

    def gen_month_mean(self):
        return self.month.groupby("id").mean()[
            [
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
        ]

    def gen_day(self, user, with_aug):
        day = pd.concat([DayData(False, with_aug).df, DayData(True).df]).join(user)

        day.index.name = "id"

        # NA
        day.loc[
            (day.kwh_pap_r1.isna())
            | (day.kwh_pap_r2.isna())
            | (day.kwh_pap_r3.isna())
            | (day.kwh_pap_r4.isna())
            | (day.kwh.isna()),
            ["kwh_pap_r1", "kwh_pap_r2", "kwh_pap_r3", "kwh_pap_r4", "kwh"],
        ] = 0

        # percent
        day["kwh_cal"] = (
            day.kwh_pap_r1 + day.kwh_pap_r2 + day.kwh_pap_r3 + day.kwh_pap_r4
        )
        day["pr2"] = day.kwh_pap_r2 / (day.kwh_cal + 0.00000001)
        day["pr3"] = day.kwh_pap_r3 / (day.kwh_cal + 0.00000001)
        day["pr4"] = day.kwh_pap_r4 / (day.kwh_cal + 0.00000001)

        # day minus
        day["2_3"] = np.abs(day.pr2 - day.pr3)
        day["2_4"] = np.abs(day.pr2 - day.pr4)
        day["3_4"] = np.abs(day.pr3 - day.pr4)
        day["daycv"] = day[["kwh_pap_r2", "kwh_pap_r3", "kwh_pap_r4"]].std(axis=1) / (
            day[["kwh_pap_r2", "kwh_pap_r3", "kwh_pap_r4"]].mean(axis=1) + 1e-5
        )

        day["type"] = 0
        day.loc[day.rq.isin(C.days_before), "type"] = -1
        day.loc[day.rq.isin(C.days_after), "type"] = 1
        day.loc[day.rq.isin(C.days_other), "type"] = 2

        day = day[day.rq != "2021-01-05"]

        return day

    def gen_day_mean(self):
        features = [
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

        mean = self.day.groupby("id").mean()[features]
        std = self.day.groupby("id").std()[features]

        mean1 = self.day[self.day.type == 0].groupby("id").mean()[features]
        mean2 = self.day[self.day.type != 0].groupby("id").mean()[features]

        std1 = self.day[self.day.type == 0].groupby("id").std()[features]
        std2 = self.day[self.day.type != 0].groupby("id").std()[features]

        mean_1_2_diff = mean1 - mean2
        mean_1_2_diff.columns = [f"{i}_holiday_diff" for i in features]

        kwh_features = [
            "kwh",
            "kwh_cal",
            "kwh_pap_r2",
            "kwh_pap_r3",
            "kwh_pap_r4",
        ]
        holiday_cv = std1[kwh_features] / (mean1[kwh_features] + 1e-5)
        holiday_cv.columns = [f"{i}_holiday_cv" for i in kwh_features]

        workday_cv = std2[kwh_features] / (mean2[kwh_features] + 1e-5)
        workday_cv.columns = [f"{i}_workday_cv" for i in kwh_features]

        cv = std[kwh_features] / (mean[kwh_features] + 1e-5)
        cv.columns = [f"{i}_cv" for i in kwh_features]

        return (
            mean.join(
                mean1.join(mean2, lsuffix="_holiday_mean_1", rsuffix="_holiday_mean_0")
            )
            .join(std1.join(std2, lsuffix="_holiday_std_1", rsuffix="_holiday_std_0"))
            .join(mean_1_2_diff)
            .join(holiday_cv)
            .join(workday_cv)
            .join(cv)
        )

    def gen_holiday_cv(self):

        df = self.day.drop(
            columns=[
                "kwh",
                "kwh_rap",
                "   ",
                "PRC_NAME",
                "CONTRACT_CAP",
                "RUN_CAP",
                "SHIFT_NO",
                "BUILD_DATE",
                "CANCEL_DATE",
                "CHK_CYCLE",
                "LAST_CHK_DATE",
                "TMP_NAME",
                "TMP_DATE",
            ]
        ).reset_index()

        df["hol_n"] = gen_day_hol_n(df)

        cols = ["kwh_pap_r2", "kwh_pap_r3", "kwh_pap_r4", "kwh_cal"]
        holcv = df.groupby(["id", "hol_n"])[cols].std() / (
            df.groupby(["id", "hol_n"])[cols].mean() + 1e-5
        )
        holcv_all = holcv.rename(
            columns={
                "kwh_pap_r1": "cv1_all",
                "kwh_pap_r2": "cv2_all",
                "kwh_pap_r3": "cv3_all",
                "kwh_pap_r4": "cv4_all",
                "kwh_cal": "cv_all",
            }
        )

        holcv = df[df.type == 0].groupby(["id", "hol_n"])[cols].std() / (
            df[df.type == 0].groupby(["id", "hol_n"])[cols].mean() + 1e-5
        )
        holcv_holiday = holcv.rename(
            columns={
                "kwh_pap_r1": "cv1_holiday",
                "kwh_pap_r2": "cv2_holiday",
                "kwh_pap_r3": "cv3_holiday",
                "kwh_pap_r4": "cv4_holiday",
                "kwh_cal": "cv_holiday",
            }
        )

        holcv = df[df.type != 0].groupby(["id", "hol_n"])[cols].std() / (
            df[df.type != 0].groupby(["id", "hol_n"])[cols].mean() + 1e-5
        )
        holcv_workday = holcv.rename(
            columns={
                "kwh_pap_r1": "cv1_workday",
                "kwh_pap_r2": "cv2_workday",
                "kwh_pap_r3": "cv3_workday",
                "kwh_pap_r4": "cv4_workday",
                "kwh_cal": "cv_workday",
            }
        )

        holcv = pd.concat(
            [holcv_all, holcv_holiday, holcv_workday], axis=1
        ).reset_index()

        holcv["isminer"] = "unknown"
        holcv.loc[holcv.id.isin(df[df.IS_FLAG == 0].id.unique()), "isminer"] = "normalt"
        holcv.loc[holcv.id.isin(C.true_ids), "isminer"] = "miner"
        # miner in train data
        holcv.loc[holcv.id.isin(C.minerids), "isminer"] = "minert"
        holcv.loc[holcv.id.isin(C.false_ids), "isminer"] = "normal"

        return holcv.groupby(["isminer", "id"]).mean().drop("hol_n", axis=1)
