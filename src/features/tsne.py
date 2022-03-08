import pandas as pd


class TSNE_filter():
    
    # 日&月&用户：按 x > 0 分割，提交成绩：
    def rule1(self, ids):
        df = pd.read_csv('dataset/tsne_all.csv')
        if len(ids) > 0:
            df = df[df.ID.isin(ids)]
        return df[df.x > 0].ID.values

    # 日&月：按 y < 50 分割，提交成绩：
    def rule2(self, ids):
        df = pd.read_csv('dataset/tsne_day_month.csv')
        if len(ids) > 0:
            df = df[df.id.isin(ids)]
        return df[df.y < 50].id.values


    # 日：按 y < 0 分割，提交成绩：
    def rule3(self, ids):
        df = pd.read_csv('dataset/tsne_day.csv')
        if len(ids) > 0:
            df = df[df.id.isin(ids)]
        return df[df.y < 0].id.values