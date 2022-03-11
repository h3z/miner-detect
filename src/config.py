from pathlib import Path
import pandas as pd

minerids = [329833811, 2294741747, 2654985038, 2674563661, 2695447713, 1998335717, 2482675592, 2614556942, 2816697143, 2825771263]

dataset_root = Path('/home/yanhuize/miner/dataset/train')


test_dataset_root = Path('/home/yanhuize/miner/dataset/test')
labelsdata = dataset_root / 'labels.csv'
test_daydata = test_dataset_root / '测试组_比特币挖矿_日用电明细（20211217）.csv'
test_monthdata = test_dataset_root / '测试组_比特币挖矿_月用电明细（20211217）.csv'


holiday_features = [
    # 'kwh_holiday_1',
    'kwh_cal_holiday_1',
    'kwh_pap_r2_holiday_1',
    'kwh_pap_r3_holiday_1',
    'kwh_pap_r4_holiday_1',
    'pr2_holiday_1',
    'pr3_holiday_1',
    'pr4_holiday_1',
    '2_3_holiday_1',
    '2_4_holiday_1',
    '3_4_holiday_1',
    'daycv_holiday_1',
    # 'kwh_holiday_0',
    'kwh_cal_holiday_0',
    'kwh_pap_r2_holiday_0',
    'kwh_pap_r3_holiday_0',
    'kwh_pap_r4_holiday_0',
    'pr2_holiday_0',
    'pr3_holiday_0',
    'pr4_holiday_0',
    '2_3_holiday_0',
    '2_4_holiday_0',
    '3_4_holiday_0',
    'daycv_holiday_0',
    # 'kwh_holiday_diff',
    'kwh_cal_holiday_diff',
    'kwh_pap_r2_holiday_diff',
    'kwh_pap_r3_holiday_diff',
    'kwh_pap_r4_holiday_diff',
    'pr2_holiday_diff',
    'pr3_holiday_diff',
    'pr4_holiday_diff',
    '2_3_holiday_diff',
    '2_4_holiday_diff',
    '3_4_holiday_diff',
    'daycv_holiday_diff']

days_other = ['2020-02-02',
              '2020-02-03',
              '2020-02-04',
              ]

days_before = ['2020-01-22',
               '2020-01-23',
               '2020-04-02',
               '2020-04-03',
               '2020-04-29',
               '2020-04-30',
               '2020-06-23',
               '2020-06-24',
               '2020-09-29',
               '2020-09-30',
               '2020-12-30',
               '2020-12-31',
               '2021-02-09',
               '2021-02-10',
               '2021-04-01',
               '2021-04-02',
               '2021-04-29',
               '2021-04-30',
               '2021-06-10',
               '2021-06-11',
               '2021-09-17',
               '2021-09-18',
               '2021-09-29',
               '2021-09-30',
               ]
days_after = ['2020-01-31',
              '2020-02-01',
              '2020-04-07',
              '2020-04-08',
              '2020-05-06',
              '2020-05-07',
              '2020-06-28',
              '2020-06-29',
              '2020-10-09',
              '2020-10-10',
              '2021-01-04',
              '2021-01-05',
              '2021-02-18',
              '2021-02-19',
              '2021-04-06',
              '2021-04-07',
              '2021-05-06',
              '2021-05-07',
              '2021-06-15',
              '2021-06-16',
              '2021-09-22',
              '2021-09-23',
              '2021-10-08',
              '2021-10-09',
              ]
days = ['2020-01-24',
        '2020-01-25',
        '2020-01-26',
        '2020-01-27',
        '2020-01-28',
        '2020-01-29',
        '2020-01-30',
        '2020-04-04',
        '2020-04-05',
        '2020-04-06',
        '2020-05-01',
        '2020-05-02',
        '2020-05-03',
        '2020-05-04',
        '2020-05-05',
        '2020-06-25',
        '2020-06-26',
        '2020-06-27',
        '2020-10-01',
        '2020-10-02',
        '2020-10-03',
        '2020-10-04',
        '2020-10-05',
        '2020-10-06',
        '2020-10-07',
        '2020-10-08',
        '2021-01-01',
        '2021-01-02',
        '2021-01-03',
        '2021-02-11',
        '2021-02-12',
        '2021-02-13',
        '2021-02-14',
        '2021-02-15',
        '2021-02-16',
        '2021-02-17',
        '2021-04-03',
        '2021-04-04',
        '2021-04-05',
        '2021-05-01',
        '2021-05-02',
        '2021-05-03',
        '2021-05-04',
        '2021-05-05',
        '2021-06-12',
        '2021-06-13',
        '2021-06-14',
        '2021-09-19',
        '2021-09-20',
        '2021-09-21',
        '2021-10-01',
        '2021-10-02',
        '2021-10-03',
        '2021-10-04',
        '2021-10-05',
        '2021-10-06',
        '2021-10-07',
        ]


def filter_features(cols, t=None, features=None):
    if features is not None:
        r = features
    elif t == 'day':
        r = day_features
    elif t == 'month':
        r = month_features
    elif t == 'user':
        r = user_features + holiday_features

    res = []
    for i in [i for i in cols if i not in r]:
        if i not in ['label', 'IS_FLAG'] + ['ELEC_TYPE_NAME', 'VOLT_NAME', 'RUN_CAP', 'ELEC_TYPE_NAME_CODE', 'VOLT_NAME_CODE'] + ['PRC_NAME', 'CONTRACT_CAP', 'SHIFT_NO', 'BUILD_DATE', 'CANCEL_DATE', 'CHK_CYCLE', 'LAST_CHK_DATE', 'TMP_NAME', 'TMP_DATE', '   ']:
            res.append(i)
    return res


user_features = [
    # 'ELEC_TYPE_NAME',
    # 'VOLT_NAME',
    # 'RUN_CAP',
    # 'label',
    # 'kwh',
    'kwh_cal',
    'kwh_pap_r2',
    'kwh_pap_r3',
    'kwh_pap_r4',
    'pr2',
    'pr3',
    'pr4',
    '2_3',
    '2_4',
    '3_4',
    'daycv',
    'pq_f',
    'pq_g',
    'pq_p',
    'pq_z',
    'pp',
    'pf',
    'pg',
    'p_f',
    'p_g',
    'f_g',
    'monthcv',
    # 'ELEC_TYPE_NAME_CODE',
    # 'VOLT_NAME_CODE'
]

day_features = [

    # 'VOLT_NAME',
    # 'kwh',
    'kwh_pap_r1',
    'kwh_pap_r2',
    'kwh_pap_r3',
    'kwh_pap_r4',
    'kwh_cal',
    'pr2',
    'pr3',
    'pr4',
    '2_3',
    '2_4',
    '3_4',
    'daycv'

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
    'pq_f',
    'pq_g',
    'pq_p',
    'pq_z',
    'pf',
    'pp',
    'pg',
    'p_f',
    'p_g',
    'f_g',
    'monthcv',

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


def daydata(submit):
    if submit:
        df = pd.read_csv(test_daydata)
    else:
        df = pd.read_csv(dataset_root / 'day.csv')
        # df = pd.read_csv(dataset_root / 'day_fillna_0.csv')
    return df


def monthdata(submit):
    if submit:
        df = pd.read_csv(test_monthdata)
    else:
        df = pd.read_csv(dataset_root / 'month.csv')
    return df.fillna(0)


def usersdata(submit):
    if submit:
        df = pd.read_csv(test_dataset_root / 'testusers.csv')
    else:
        df = pd.read_csv(dataset_root / 'users.csv')
    return df.fillna(0)
