import config as C
import pandas as pd
# from models.gru_4_month import train, predict
# from models.linear_4_month import train, predict
from models import seq2seq
from features.DayData import DayData
from features.MonthData import MonthData
from features.UsersData import UsersData

def score(ids):
    tp = len([i for i in ids if i in C.minerids])
    fp = len([i for i in ids if i not in C.minerids])
    fn = len([i for i in C.minerids if i not in ids])

    p = tp / (tp + fp)
    r = tp / (fp + fn)

    return 2*p*r/(p+r)


# df = UsersData().df
# ids = df.ID.values

# ids = MonthData().rule1_month_percent(ids)
# ids = MonthData().rule2_22month_total_power(ids)

# ids = UsersData().rule1_small_cap(ids)

# print(score(ids))
# print(len(ids))

# model = train()




# train day power use seq2seq
# seq2seq.train()

# train month power use seq2seq
seq2seq.MonthDataEncode().train()
encoder = seq2seq.MonthDataEncode('/home/yanhuize/miner/src/ttenc_train_test_month_32_4feature.torch')
train_month_df = MonthData().df
test_month_df = MonthData(True).df

# train day data use seq2seq
# seq2seq.train()

# predict()
# res = predict(test_month, model)

