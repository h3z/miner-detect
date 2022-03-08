import config as C
import pandas as pd

'''
id          	用户唯一标识
elec_type_name	用电类别名称 8 ['乡村居民生活用电' '城镇居民生活用电' '普通工业' '非居民照明' '学校教学和学生生活用电' '商业用电' '居民生活用电' '非工业']
volt_name	    电压等级名称 3 ['交流220V' '交流380V' '交流10kV'] 
prc_name	    电价名称 2 ['居民合表电价(不满1千伏）' '居民合表电价(1-10千伏）']
contract_cap	合同容量
run_cap	        运行容量
shift_no    	生产班次 5 [ 1. nan  4.  2.  3.]
build_date    	立户日期
cancel_date    	销户日期 5 [nan '2021/10/11 9:44' '2021/10/11 9:38' '2021/10/27 15:32' '2021/10/18 11:22']
chk_cycle    	检查周期 18 [240.  24.  12.  60.  47.  16.   6.  48.  25.  36.  42.   3.  72. 120.   1.   9.  10.  nan]
last_chk_date	上次检查日期
tmp_name	    临时用电标志 1 ['非临时用电']
tmp_date	    临时用电到期日期 2 [nan '2015/9/11']
IS_FLAG	    	是否挖矿 2 [0 1]
'''

class UsersData:
    def __init__(self, submit=False) -> None:
        # df.rq = pd.to_datetime(df.rq)
        self.df = C.usersdata(submit)

        self.SMALL_CONTRAT_CAP = 100


    def rule1_small_cap(self, ids):
        df = self.df
        if len(ids) > 0:
            df = self.df[self.df.ID.isin(ids)]
        

        return df[df.CONTRACT_CAP < self.SMALL_CONTRAT_CAP].ID.values




