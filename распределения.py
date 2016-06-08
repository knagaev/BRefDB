
Измерения: контора, матч, вариант, время
Значение: вероятность

По вариантам (фильтруем один вариант).
Распределение матчей по близости к результату (по последнему временному отсчету)   


import pymssql
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import grid_search
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.preprocessing import scale, StandardScaler
from sklearn.metrics import roc_auc_score, make_scorer
import time
import datetime
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')


conn = pymssql.connect('.\\SQLEXPRESS', 'BB_miner', 'BB_3817_miner', "BRefDB")

df_lines = pd.read_sql('select line_ID, house_ref, match_ref, TS_ref, line_value, line_increment, snapshot_time, is_it_starting, RTV_ref, time_increment from Lines', conn, index_col='line_ID')

# фильтруем первый вариант 
df_first_win_lines = df_lines[df_lines['RTV_ref'] == 1]

# превращаем is_it_starting в отсчет за секунду
df_starting_lines = df_first_win_lines[df_first_win_lines['is_it_starting'] == 1]
df_not_starting_lines = df_first_win_lines[df_first_win_lines['is_it_starting'] == 0]

merge(df_starting_lines, df_not_starting_lines, on=['match_ref', 'TS_ref'])

df_bets_by_bet = df_first_win_lines.pivot_table(index=['TS_ref'], columns=['RTV_ref', 'house_ref'], values=['prob_value'])
