from pathlib import Path
import pandas as pd
import random
from features.MonthData import MonthData
from features.DayData import DayData
from features.UsersData import UsersData
import models.seq2seq as TT
import config  as C
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import numpy as np

def preprocessing(df):
    df = TT.count_filter(df)
    return df[['kwh', 'id']]

def all_traindata():
    df = DayData().df
    return preprocessing(df)

def all_testdata():
    df = DayData(True).df
    return preprocessing(df)


def filtered_data(istest):
    ids = []
    # 1/3
    ids = MonthData(istest).rule1_month_percent(ids)
    # 总电量
    ids = MonthData(istest).rule2_22month_total_power(ids)

    df = DayData(istest).df
    df = df[df.id.isin(ids)]
    return preprocessing(df)

def filtered_testdata():
    return filtered_data(True)

def filtered_traindata():
    return filtered_data(False)