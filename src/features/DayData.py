import config as C
import pandas as pd
import numpy as np
'''
id	用户唯一标识
rq	日期
kwh	当日抄见电量
kwh_rap	昨日反向电量
kwh_pap_r1	昨日正向费率1电量
kwh_pap_r2	昨日正向费率2电量
kwh_pap_r3	昨日正向费率3电量
kwh_pap_r4	昨日正向费率4电量
'''

class DayData:
    def __init__(self, submit=False) -> None:
        df = C.daydata(submit)
        
        df.rq = pd.to_datetime(df.rq)

        kwh = df.kwh + 0.0000001
        df['kwh_cal'] = df.kwh_pap_r1 + df.kwh_pap_r2 + df.kwh_pap_r3 + df.kwh_pap_r4
        df['pr2'] = np.abs(df.kwh_pap_r2 / (df.kwh_cal + 0.00000001))
        df['pr3'] = np.abs(df.kwh_pap_r3 / (df.kwh_cal + 0.00000001))
        df['pr4'] = np.abs(df.kwh_pap_r4 / (df.kwh_cal + 0.00000001))
        self.df = df