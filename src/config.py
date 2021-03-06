from pathlib import Path
import pandas as pd

minerids = [
    329833811,
    2294741747,
    2654985038,
    2674563661,
    2695447713,
    1998335717,
    2482675592,
    2614556942,
    2816697143,
    2825771263,
]

dataset_root = Path("/home/yanhuize/miner/dataset/train")


test_dataset_root = Path("/home/yanhuize/miner/dataset/test")
labelsdata = dataset_root / "labels.csv"
test_daydata = test_dataset_root / "测试组_比特币挖矿_日用电明细（20211217）.csv"
test_monthdata = test_dataset_root / "测试组_比特币挖矿_月用电明细（20211217）.csv"


# ids_5_1 = {179558804, 1688684718, 2099471710, 2323237963, 2576316686}
# ids_2_1 = {2323237963, 2576316686}


true_ids = (
    {
        179458306,
        1606708811,
        2071313507,
        2427050072,
        179569820,
        2186749200,
        2212416005,
        2256064355,
        2347718610,
        2576321385,
    }
    | {
        2238809293,
        2741872006,
        2817362052,
        2825175309,
        
        2852503463,
        2533183958,
        2319973783,
        1916407803,
        855996491,
    }
    | {2717225077, 2445049876, 2172970175}
    | {2523401557}
    | {2471562086, 179418058, 362400993}
    | {2347718608, 2759232590, 2496032641}
    | {1912367373, 2745781539}
    | {2624797677}
    | {1862376457}
    | {2540517219}
    | {2212577893}
    | {179547052}
    | {2323237963}
)


false_ids_3_17_1 = {
    889636423,
    1541520478,
    1823990276,
    1868915135,
    1990279480,
    2190112718,
    2299338729,
    2306466642,
    2327122022,
    2337066378,
    2520770436,
    2639515021,
    2717225078,
}
false_ids = (
    {2245541933, 2268548573, 2755128229, 179519109, 2271232929, 2759317616, 2717225084}
    | {2451165982}
    | {2251440776, 179433516, 887227796}
    | {2479963778, 2535198432}
    | {2602819207}
    | {2194274407, 2115712219, 179569085, 2044965214, 1429892107}
    | {2784950026, 179571678}
    | {440775153, 179506462, 874996400}
    | false_ids_3_17_1
    | {179418259, 179536619, 179570074, 1862415085, 179517153, 2494025452}
    | {2271557820, 2442626095, 2535198442, 1480738101}
    | {
        179407055,
        179420554,
        179421630,
        179457448,
        179516332,
        179517418,
        179588734,
        400849958,
        880030471,
        1485790704,
        1862415085,
        2045098996,
        2271557820,
        2326835483,
        2689150327,
        179474385,
        179574569,
        1564847092,
        2181712126,
        2269728686,
    }
    | {2098712195, 2533183961}
)


day_origin = [
    "kwh",
    "kwh_pap_r1",
    "kwh_pap_r2",
    "kwh_pap_r3",
    "kwh_pap_r4",
]

holiday_features = [
    # 'kwh_holiday_1',
    "kwh_cal_holiday_1",
    "kwh_pap_r2_holiday_1",
    "kwh_pap_r3_holiday_1",
    "kwh_pap_r4_holiday_1",
    "pr2_holiday_1",
    "pr3_holiday_1",
    "pr4_holiday_1",
    "2_3_holiday_1",
    "2_4_holiday_1",
    "3_4_holiday_1",
    "daycv_holiday_1",
    # 'kwh_holiday_0',
    "kwh_cal_holiday_0",
    "kwh_pap_r2_holiday_0",
    "kwh_pap_r3_holiday_0",
    "kwh_pap_r4_holiday_0",
    "pr2_holiday_0",
    "pr3_holiday_0",
    "pr4_holiday_0",
    "2_3_holiday_0",
    "2_4_holiday_0",
    "3_4_holiday_0",
    "daycv_holiday_0",
    # 'kwh_holiday_diff',
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

days_other = [
    "2020-02-02",
    "2020-02-03",
    "2020-02-04",
]

days_before = [
    "2020-01-22",
    "2020-01-23",
    "2020-04-02",
    "2020-04-03",
    "2020-04-29",
    "2020-04-30",
    "2020-06-23",
    "2020-06-24",
    "2020-09-29",
    "2020-09-30",
    "2020-12-30",
    "2020-12-31",
    "2021-02-09",
    "2021-02-10",
    "2021-04-01",
    "2021-04-02",
    "2021-04-29",
    "2021-04-30",
    "2021-06-10",
    "2021-06-11",
    "2021-09-17",
    "2021-09-18",
    "2021-09-29",
    "2021-09-30",
]
days_after = [
    "2020-01-31",
    "2020-02-01",
    "2020-04-07",
    "2020-04-08",
    "2020-05-06",
    "2020-05-07",
    "2020-06-28",
    "2020-06-29",
    "2020-10-09",
    "2020-10-10",
    "2021-01-04",
    "2021-01-05",
    "2021-02-18",
    "2021-02-19",
    "2021-04-06",
    "2021-04-07",
    "2021-05-06",
    "2021-05-07",
    "2021-06-15",
    "2021-06-16",
    "2021-09-22",
    "2021-09-23",
    "2021-10-08",
    "2021-10-09",
]
days = [
    "2020-01-24",
    "2020-01-25",
    "2020-01-26",
    "2020-01-27",
    "2020-01-28",
    "2020-01-29",
    "2020-01-30",
    "2020-04-04",
    "2020-04-05",
    "2020-04-06",
    "2020-05-01",
    "2020-05-02",
    "2020-05-03",
    "2020-05-04",
    "2020-05-05",
    "2020-06-25",
    "2020-06-26",
    "2020-06-27",
    "2020-10-01",
    "2020-10-02",
    "2020-10-03",
    "2020-10-04",
    "2020-10-05",
    "2020-10-06",
    "2020-10-07",
    "2020-10-08",
    "2021-01-01",
    "2021-01-02",
    "2021-01-03",
    "2021-02-11",
    "2021-02-12",
    "2021-02-13",
    "2021-02-14",
    "2021-02-15",
    "2021-02-16",
    "2021-02-17",
    "2021-04-03",
    "2021-04-04",
    "2021-04-05",
    "2021-05-01",
    "2021-05-02",
    "2021-05-03",
    "2021-05-04",
    "2021-05-05",
    "2021-06-12",
    "2021-06-13",
    "2021-06-14",
    "2021-09-19",
    "2021-09-20",
    "2021-09-21",
    "2021-10-01",
    "2021-10-02",
    "2021-10-03",
    "2021-10-04",
    "2021-10-05",
    "2021-10-06",
    "2021-10-07",
]


def filter_features(cols, t=None, features=None):
    if features is not None:
        r = features
    elif t == "day":
        r = day_features
    elif t == "month":
        r = month_features
    elif t == "user":
        r = user_features + holiday_features

    res = []
    for i in [i for i in cols if i not in r]:
        if i not in ["label", "IS_FLAG"] + [
            "ELEC_TYPE_NAME",
            "VOLT_NAME",
            "RUN_CAP",
            "ELEC_TYPE_NAME_CODE",
            "VOLT_NAME_CODE",
        ] + [
            "PRC_NAME",
            "CONTRACT_CAP",
            "SHIFT_NO",
            "BUILD_DATE",
            "CANCEL_DATE",
            "CHK_CYCLE",
            "LAST_CHK_DATE",
            "TMP_NAME",
            "TMP_DATE",
            "   ",
        ]:
            res.append(i)
    return res


user_features = [
    # 'ELEC_TYPE_NAME',
    # 'VOLT_NAME',
    # 'RUN_CAP',
    # 'label',
    # 'kwh',
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
    # 'ELEC_TYPE_NAME_CODE',
    # 'VOLT_NAME_CODE'
]

day_features = [
    # 'VOLT_NAME',
    # 'kwh',
    "kwh_pap_r1",
    "kwh_pap_r2",
    "kwh_pap_r3",
    "kwh_pap_r4",
    "kwh_cal",
    "pr2",
    "pr3",
    "pr4",
    "2_3",
    "2_4",
    "3_4",
    "daycv"
    # possible
    # 'kwh_rap',
    # 'rq'
    # 'ym',
    # 'CONTRACT_CAP',
    # 'RUN_CAP',
    # 'ELEC_TYPE_NAME',
    # useless
    # 'PRC_NAME',
    # 'SHIFT_NO',
    # 'BUILD_DATE',
    # 'CANCEL_DATE',
    # 'CHK_CYCLE',
    # 'LAST_CHK_DATE',
]

month_features = [
    # strong
    # 'VOLT_NAME',
    "pq_f",
    "pq_g",
    "pq_p",
    "pq_z",
    "pf",
    "pp",
    "pg",
    "p_f",
    "p_g",
    "f_g",
    "monthcv",
    # possible
    # 'ym',
    # 'CONTRACT_CAP',
    # 'RUN_CAP',
    # 'ELEC_TYPE_NAME',
    # useless
    # 'PRC_NAME',
    # 'SHIFT_NO',
    # 'BUILD_DATE',
    # 'CANCEL_DATE',
    # 'CHK_CYCLE',
    # 'LAST_CHK_DATE',
]


# def custom_aug(df):
#     d = df[df.index.isin(minerids)]
#     idmax = 2880712108

#     newdfs = []
#     for i in range(1, 50):
#         newd = d.copy()
#         newd = newd.reset_index()
#         newd.id = newd.id + idmax * i
#         newd = newd.set_index("id")
#         newdfs.append(newd)

#     res = pd.concat(newdfs + [df])
#     print("after", res.shape)
#     return res


def custom_aug(d):
    print("before", d.shape)
    # idmax = d.index.max()
    # train & test
    idmax = 2880712108

    scale = 2
    features = {
        "pq_f",
        "pq_p",
        "pq_g",
        "pq_z",
        "kwh_pap_r4",
        "kwh_pap_r3",
        "kwh_pap_r2",
        "kwh_pap_r1",
        "kwh",
    }

    features &= set(d.columns.values)
    features = list(features)

    p = d[features] / scale
    newdfs = []
    for idx, i in enumerate(range(int(-0.5 * scale), int(1 * scale))):
        newd = d.copy()

        # newd[features] = newd[features] + p * i
        # newd[features] = newd[features]
        newd = newd.reset_index()
        newd.id = newd.id + idmax * (idx + 1)
        newd = newd.set_index("id")

        newdfs.append(newd)

    res = pd.concat(newdfs + [d])
    print("after", res.shape)
    return res


def daydata(submit, withaug):
    if submit:
        df = pd.read_csv(test_daydata)
    else:
        df = pd.read_csv(dataset_root / "day.csv")
        # df = pd.read_csv(dataset_root / 'day_fillna_0.csv')
    df = df.set_index("id")
    if withaug:
        return custom_aug(df)
    return df


def monthdata(submit, withaug):
    if submit:
        df = pd.read_csv(test_monthdata)
    else:
        df = pd.read_csv(dataset_root / "month.csv")
    df = df.set_index("id")
    if withaug:
        return custom_aug(df.fillna(0))
    return df.fillna(0)


def usersdata(submit, withaug):
    if submit:
        df = pd.read_csv(test_dataset_root / "testusers.csv")
    else:
        df = pd.read_csv(dataset_root / "users.csv")

    df["id"] = df.ID
    df = df.drop(columns=["ID"])
    df = df.set_index("id")
    if withaug:
        return custom_aug(df.fillna(0))
    return df.fillna(0)
