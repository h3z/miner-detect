from features.MonthData import MonthData
from features.UsersData import UsersData
from features.tsne import TSNE_filter
import config as C
import pandas as pd


def f(tt):
    # 三分之一 & 总容量 ，0.7 左右
    df = UsersData(True).df
    print(df.shape)
    ids = df.ID.values
    ids = MonthData(True).rule1_month_percent(ids)
    ids = MonthData(True).rule2_22month_total_power(ids)

    if tt == 1:
        # 日&月&用户：按 x > 0 分割，提交成绩：
        ids = TSNE_filter().rule1(ids)

    elif tt == 2:
        # 日&月：按 y < 50 分割，提交成绩：
        ids = TSNE_filter().rule2(ids)

    else:
        # 日：按 y < 0 分割，提交成绩：
        ids = TSNE_filter().rule3(ids)


######
    # 330V 过滤
    

    print(len(ids))
    res = []
    for id in UsersData(True).df.ID.values:
        res.append([id, int(id in ids)])

    df = pd.DataFrame(res, columns=['id', 'label'])

    df.to_csv(f'submit{tt}.csv', index=False)


if __name__ == '__main__':
    f(1)
    f(2)
    f(3)