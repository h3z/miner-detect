import sys
sys.path.append('../src')
import itertools
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from pathlib import Path
import random
from xgboost import plot_tree
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import config as C
from features.UserJoin import submit, diff, plt_month, plt_day, load_ids, info, infot
from features.UserJoin import UserJoin
import pandas as pd
import xgboost as xgb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


ds = UserJoin()

user = ds.train[C.user_features]
monthjoin = ds.month.join(user, rsuffix='_mean')

mean_cols = []
for i in monthjoin.columns:
    if 'mean' in i:
        monthjoin[i.replace('mean', 'sub_mean')] = monthjoin[i.replace('_mean', '')] - monthjoin[i]
        mean_cols.append(i)

monthjoin = monthjoin.drop(columns=mean_cols)

monthjoin_features = [
    'pq_f',
    'pq_g',
    'pq_p',
    'pq_z',
    'pf',
    'pg',
    'pp',
    'p_f',
    'p_g',
    'f_g',
    'monthcv',
    # 'pq_f_mean',
    # 'pq_g_mean',
    # 'pq_p_mean',
    # 'pq_z_mean',
    # 'pp_mean',
    # 'pf_mean',
    # 'pg_mean',
    # 'p_f_mean',
    # 'p_g_mean',
    # 'f_g_mean',
    # 'monthcv_mean',

    'pq_f_sub_mean',
    'pq_g_sub_mean',
    'pq_p_sub_mean',
    'pq_z_sub_mean',
    'pp_sub_mean',
    'pf_sub_mean',
    'pg_sub_mean',
    'p_f_sub_mean',
    'p_g_sub_mean',
    'f_g_sub_mean',
    'monthcv_sub_mean',
]

def exp(features, df, params):
    x, x_val = df.loc[df.label != 'test', features], df.loc[df.label == 'test', features]
    y = df.loc[df.label != 'test', 'IS_FLAG']
    model = xgb.XGBClassifier(verbosity=0, **params)
    model.fit(x, y)
    pred_y = model.predict(x)
    # print((pred_y != y).sum())

    y_val = model.predict(x_val)
    pred_val = pd.DataFrame({'id': x_val.index.values, 'pred': y_val}).groupby('id').sum()
    pred_y = pd.DataFrame({'id': x.index.values, 'ym': df.loc[df.label != 'test'].ym.values, 'pred': pred_y, 'label': y.values})
    return pred_y, pred_val


def today_exp(params, aug=False, join=False):
    params['use_label_encoder'] = False

    pred_y, pred_val = exp(monthjoin_features, monthjoin, params)

    res = []
    for i in range(22):
        ids = set(pred_val[pred_val.pred > i].index.values)
        res.append([i, len(ids & C.true_ids), len(ids & C.false_ids), len(ids - C.true_ids - C.false_ids), len(ids), str(ids)])
    return res


if __name__ == '__main__':
    params = {
        'max_depth': [3, 5, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 500],
        'colsample_bytree': [0.3, 0.4, 0.6, 0.8, 1],
        'subsample': [0.4, 0.8, 1]
    }

    dfs = []
    for max_depth, learning_rate, n_estimators, colsample_bytree, subsample in tqdm(list(itertools.product(params['max_depth'],
                                                                                                            params['learning_rate'],
                                                                                                            params['n_estimators'],
                                                                                                            params['colsample_bytree'],
                                                                                                            params['subsample']))):
        p = {
            'max_depth': max_depth,
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'colsample_bytree': colsample_bytree,
            'subsample': subsample,
        }
        res = today_exp(p)
        for i in res:
            p['gt'] = i[0]
            p['total'] = i[4]
            p['right_miner'] = i[1]
            p['wrong_miner'] = i[2]
            p['unknown'] = i[3]
            p['ids'] = i[5]

            dfs.append(pd.DataFrame(p, index=[0]).drop(columns=['use_label_encoder']))

        df = pd.concat(dfs)
        df.to_csv('test_withjoin.csv', index=False)