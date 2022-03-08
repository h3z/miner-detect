from pathlib import Path
import pandas as pd

minerids = [329833811, 2294741747, 2654985038, 2674563661, 2695447713, 1998335717, 2482675592, 2614556942, 2816697143, 2825771263]

dataset_root = Path('/home/yanhuize/miner/dataset/train')


test_dataset_root = Path('/home/yanhuize/miner/dataset/test')
labelsdata = dataset_root / 'labels.csv'
test_daydata = test_dataset_root / '测试组_比特币挖矿_日用电明细（20211217）.csv'
test_monthdata = test_dataset_root / '测试组_比特币挖矿_月用电明细（20211217）.csv'


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
