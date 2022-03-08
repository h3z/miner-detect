import config as C
import pandas as pd
from dateutil.relativedelta import relativedelta

'''
id	用户唯一标识
ym	电费年月
pq_f	峰电量
pq_g	谷电量
pq_p	平电量
pq_z	总电量

没有 NA
'''


class MonthData:

    def __init__(self, submit=False) -> None:
        df = C.monthdata(submit)
        
        t = 0.05
        self.MIN = 0.33 - t
        self.MAX = 0.33 + t
        self.COUNT = 15
        self.POWER_SUM = 50000
        
        self.R3_MONTHS = 6

        df.ym = pd.to_datetime(df.ym, format='%Y%m')

        pq_z = df.pq_z + 0.0000001
        df['pf'] = df.pq_f / pq_z
        df['pg'] = df.pq_g / pq_z
        df['pp'] = df.pq_p / pq_z

        self.df = df

    def rule1_month_percent(self, ids):
        df = self.df
        if len(ids) > 0:
            df = self.df[self.df.id.isin(ids)]

        df['count'] = (df.pf > self.MIN) & (df.pf < self.MAX) & (df.pg > self.MIN) & (df.pg < self.MAX) & (df.pp > self.MIN) & (df.pp < self.MAX)
        tmp = df.groupby('id')['count'].sum()
        return tmp[tmp > self.COUNT].index.values

    def rule3_6month_std(self, ids):
        df = self.df
        if len(ids) > 0:
            df = self.df[self.df.id.isin(ids)]

        stds = []
        start_month = df.ym.min()
        end_month = start_month + relativedelta(months=self.R3_MONTHS)
        while end_month <= df.ym.max():
            std = df[(df.ym <= end_month) & (df.ym >= start_month)].groupby('id')[['pp', 'pf', 'pg']].std()
            stds.append(std)
            start_month += relativedelta(months=1)
            end_month += relativedelta(months=1)

        return ids, stds


    def rule2_22month_total_power(self, ids):
        df = self.df
        if len(ids) > 0:
            df = self.df[self.df.id.isin(ids)]

        tmp = df.groupby('id')['pq_z'].sum()
        return tmp[tmp > self.POWER_SUM].index.values

    def rules(self):
        ids = self.rule1_month_percent([])
        ids = self.rule2_22month_total_power(ids)
        # ids, stds = self.rule3_6month_std(ids)
        return ids


    



