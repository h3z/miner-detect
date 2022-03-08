from features.MonthData import MonthData
from features.UsersData import UsersData
from features.DayData import DayData
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import config as C
import plotly.express as px


def plt_day(allids, df, cols=['pr2', 'pr3', 'pr4'], x='rq'):

    dct = {}
    for i, v in enumerate(df[x].sort_values().unique()):
        dct[v] = i


    if type(allids) is str:
        t = pd.read_csv(allids)
        allids = t[t.label == 1].id.unique()

    if 'id' not in df.columns:
        df['id'] = df.index

    def f(ids):
        print(ids)
        d = df[df.id.isin(ids)]
        d = d.replace({x: dct})
        fig = px.line(d, x=x, y=cols, facet_col="id", facet_col_wrap=4, height=(int((len(ids)-1)/4)+1)*300, range_y=(0,1))
        fig.show()

    
    for i in range(int(((len(allids) - 1) / 20) + 1)):
        if i*20+20 > len(allids):
            f(allids[i*20:])
        else:
            f(allids[i*20:i*20+20])


def plt_month(allids, df, cols=['pp', 'pf', 'pg'], x='ym'):
    if type(allids) is str:
        t = pd.read_csv(allids)
        allids = t[t.label == 1].id.unique()

    if 'id' not in df.columns:
        df['id'] = df.index

    def f(ids):
        print(ids)
        d = df[df.id.isin(ids)]
        fig = px.line(d, x=x, y=cols, facet_col="id", facet_col_wrap=4, height=((len(ids)-1)/4+1)*300, range_y=(0,1))
        fig.show()

    for i in range(int(((len(allids) - 1) / 20) + 1)):
        if i*20+20 > len(allids):
            f(allids[i*20:])
        else:
            f(allids[i*20:i*20+20])


def check_f(s, df):
    def norm_arr(s, n=2):
        return s[((s < (s.mean() + n*s.std())) & (s > (s.mean() - n*s.std())))]
    norm_arr(df[s]).hist(bins=60, alpha=.4)

    s2 = df[df.index.isin(C.minerids)][s]
    s2.hist(bins=60)
    print(s2.min(), s2.max())
    plt.axvline(x=s2.min(), c='r')
    plt.axvline(x=s2.max(), c='r')
    plt.title(s)
    plt.show()


def to_cat_code(df, col):
    df[col] = df[col].astype('category')
    df[col + '_CODE'] = df[col].cat.codes
    return df


def scale(df):
    scale = 0.05
    std_max = 0.05
    return df[(df.pp < 0.333 + scale) & (df.pp > 0.333 - scale) & (df.pf < 0.333 + scale) & (df.pf > 0.333 - scale) & (df.pg < 0.333 + scale) & (df.pg > 0.333 - scale)]


def to_path(f):
    return f'submit_csv/{f}'


def load_ids(f):
    d = pd.read_csv(to_path(f))
    return set(d[d.label == 1].id.values)

def diff(f1, f2):
    d1 = pd.read_csv(to_path(f1))
    d2 = pd.read_csv(to_path(f2))

    s1 = set(d1[d1.label == 1].id.values)
    s2 = set(d2[d2.label == 1].id.values)

    print('in s1 not s2', len(s1-s2))
    print('in s2 not s1', len(s2-s1))
    print('inner', len(s2 & s1))
    return s1 & s2


def submit(ids, f):
    res = []
    for id in UsersData(True).df.ID.values:
        res.append([id, int(id in ids)])

    df = pd.DataFrame(res, columns=['id', 'label'])
    print(df.label.sum())
    df.to_csv(to_path(f), index=False)
    return df


class UserJoin():
    def __init__(self) -> None:
        user, usert = UsersData().df, UsersData(True).df
        user['label'] = user['IS_FLAG'].astype('str')
        usert['label'] = 'test'
        user = pd.concat([user, usert])

        self.day = self.gen_day(user)
        self.month = self.gen_month(user)
        self.day_mean = self.gen_day_mean()
        self.month_mean = self.gen_month_mean()

        # join to ds
        train = user.set_index('ID')[['ELEC_TYPE_NAME', 'VOLT_NAME', 'RUN_CAP', 'label']] \
            .join(self.day_mean) \
            .join(self.month_mean) \
            .dropna()
        train.index.name = 'id'

        train = to_cat_code(train, 'ELEC_TYPE_NAME')
        train = to_cat_code(train, 'VOLT_NAME')
        train = train[(train.pq_p >= 0) &
                      (train.pq_g >= 0) &
                      (train.pq_z >= 0)]

        self.train = self.rule1(train)
        self.train1 = self.rule2(self.train)
        self.train2 = self.rule3(self.train1)

    def rule1(self, df):
        return df[df.VOLT_NAME == "交流380V"]

    def rule2(self, df):
        return df[df.kwh > 50]

    def rule3(self, df):
        scale = 0.05
        std_max = 0.05
        return df[(df.pp < 0.333 + scale)
                  & (df.pp > 0.333 - scale)
                  & (df.pf < 0.333 + scale)
                  & (df.pf > 0.333 - scale)
                  & (df.pg < 0.333 + scale)
                  & (df.pg > 0.333 - scale)]

    def gen_month(self, user):
        month = pd.concat([MonthData().df, MonthData(True).df]) \
            .set_index('id') \
            .join(user.set_index('ID'))
        month.index.name = 'id'

        # month minus
        month['p_f'] = np.abs(month.pp - month.pf)
        month['p_g'] = np.abs(month.pp - month.pg)
        month['f_g'] = np.abs(month.pf - month.pg)
        month['monthcv'] = month[['pq_g', 'pq_p', 'pq_f']].std(axis=1) / (month[['pq_g', 'pq_p', 'pq_f']].mean(axis=1) + 1e-5)
        return month

    def gen_month_mean(self):
        return self.month.groupby('id').mean()[[
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
            'monthcv'
        ]]

    def gen_day(self, user):
        day = pd.concat([DayData().df, DayData(True).df]) \
            .set_index('id') \
            .join(user.set_index('ID'))

        day.index.name = 'id'

        # NA
        day.loc[
            (day.kwh_pap_r1.isna()) &
            (day.kwh_pap_r2.isna()) &
            (day.kwh_pap_r3.isna()) &
            (day.kwh_pap_r4.isna()) &
            (day.kwh.isna()),
            ['kwh_pap_r1', 'kwh_pap_r2', 'kwh_pap_r3', 'kwh_pap_r4', 'kwh']
        ] = 0

        # percent
        day['kwh_cal'] = day.kwh_pap_r1 + day.kwh_pap_r2 + day.kwh_pap_r3 + day.kwh_pap_r4
        day['pr2'] = day.kwh_pap_r2 / (day.kwh_cal + 0.00000001)
        day['pr3'] = day.kwh_pap_r3 / (day.kwh_cal + 0.00000001)
        day['pr4'] = day.kwh_pap_r4 / (day.kwh_cal + 0.00000001)

        # day minus
        day['2_3'] = np.abs(day.pr2 - day.pr3)
        day['2_4'] = np.abs(day.pr2 - day.pr4)
        day['3_4'] = np.abs(day.pr3 - day.pr4)
        day['daycv'] = day[['kwh_pap_r2', 'kwh_pap_r3', 'kwh_pap_r4']].std(axis=1) / (day[['kwh_pap_r2', 'kwh_pap_r3', 'kwh_pap_r4']].mean(axis=1) + 1e-5)
        return day

    def gen_day_mean(self):
        return self.day.groupby('id').mean()[[
            'kwh',
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
            'daycv'
        ]]
