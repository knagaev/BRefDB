import pymssql
import pandas as pd
import numpy as np
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn import grid_search
#from sklearn.cross_validation import KFold, cross_val_score
#from sklearn.preprocessing import scale, StandardScaler
#from sklearn.metrics import roc_auc_score, make_scorer
#import time
#import datetime
#from datetime import timedelta
#from sklearn.linear_model import LogisticRegression
import matplotlib
#matplotlib.use("qt4agg")
matplotlib.style.use('ggplot')
from matplotlib import pyplot as plt

#fig = plt.figure()
#axis = fig.add_subplot(111)

conn = pymssql.connect('.\\SQLEXPRESS', 'BB_miner', 'BB_3817_miner', "BRefDB")

df_lines = pd.read_sql('''select line_ID, house_ref, match_ref, TS_ref, line_value, line_increment, snapshot_time, is_it_starting, RTV_ref, time_increment 
                          from Lines where RTV_ref in (1, 2, 3)''', conn, index_col='line_ID')

df_match_results = pd.read_sql('''select MR_ID, RTV_ref, match_ref, actual_value, text_result
                          from Match_results where RTV_ref in (1, 2, 3)''', conn, index_col='MR_ID')

# корректировка дат для записей is_it_starting == 1
df_s_lines = df_lines[df_lines['is_it_starting'] == 1]
df_ns_lines = df_lines[df_lines['is_it_starting'] == 0]
df_first_ns_lines = df_ns_lines.loc[df_ns_lines.groupby(['match_ref', 'house_ref', 'RTV_ref'])['TS_ref'].idxmin()]
%xdel df_ns_lines

df_merged_s_ns = pd.merge(df_s_lines.reset_index(), df_first_ns_lines.reset_index(), how='inner', on=['match_ref', 'house_ref', 'RTV_ref', 'TS_ref'], suffixes=['_s', '_ns']).set_index('line_ID_s')
%xdel df_s_lines
%xdel df_first_ns_lines

ser_corr_stimes = df_merged_s_ns['snapshot_time_ns'] - pd.to_timedelta(df_merged_s_ns['time_increment_ns'], unit='m')
%xdel df_merged_s_ns

df_lines.loc[df_lines['is_it_starting'] == 1, 'snapshot_time'] = ser_corr_stimes
%xdel ser_corr_stimes

# наполнение df_lines пропущенными записями
#all_matches = df_lines['match_ref'].unique()
#all_snapshot_times = df_lines['snapshot_time'].unique()

all_houses = pd.DataFrame(df_lines['house_ref'].unique(), columns=['house_ref'])
all_houses['key'] = 1
all_rtv_refs = pd.DataFrame(df_lines['RTV_ref'].unique(), columns=['RTV_ref'])
all_rtv_refs['key'] = 1
all_house_refs = pd.merge(all_houses, all_rtv_refs, on='key')

#full_index = pd.MultiIndex.from_product(dimensions, names=['match_ref', 'house_ref', 'RTV_ref', 'snapshot_time'])
#full_index = pd.MultiIndex.from_tuples(df_lines.groupby(['match_ref', 'snapshot_time']).groups.keys(), names=['match_ref', 'snapshot_time'])

df_match_times = pd.DataFrame(df_lines.groupby(['match_ref', 'snapshot_time']).groups.keys(), columns=['match_ref', 'snapshot_time'])
df_match_times['key'] = 1

all_match_times_house_refs = pd.merge(df_match_times, all_house_refs, on='key').drop('key', axis=1)
%xdel df_match_times
%xdel all_house_refs

#sorted(all_match_times_house_refs[all_match_times_house_refs.match_ref == 1573]['snapshot_time'].unique())
#SELECT DISTINCT l.is_it_starting, l.snapshot_time FROM Lines123 l WHERE l.match_ref = 1573;
#SELECT * FROM Lines123 l WHERE l.match_ref = 1573

#df_full_lines = pd.merge(all_match_times_house_refs.reset_index(), df_lines.reset_index(), how='left', on=['match_ref', 'house_ref', 'RTV_ref', 'snapshot_time'], suffixes=['_l', '_r'])
df_full_lines = pd.merge(all_match_times_house_refs, df_lines, how='left', on=['match_ref', 'house_ref', 'RTV_ref', 'snapshot_time'], suffixes=['_l', '_r'])
df_full_lines.index.name = 'ndx'
#df_full_lines[df_full_lines.match_ref == 1573].to_csv(r'c:\work\others\Oleg\test.csv')
%xdel all_match_times_house_refs

df_full_lines.set_index(['match_ref', 'house_ref', 'RTV_ref', 'snapshot_time'], inplace=True)

df_full_lines.sort_index(inplace=True)
df_full_lines.sort_index(axis=1, inplace=True)

#df_filled_full_lines = df_full_lines.fillna(method='ffill').fillna(method='bfill')
df_filled_full_lines = df_full_lines.groupby(level=[0, 1, 2]).fillna(method='ffill').fillna(method='bfill')

df_full_lines.loc[(1573, slice(None), slice(None), pd.Timestamp('2016-03-29 08:22:00')), :]

df_filled_full_lines.loc[(1573, slice(None), 1, pd.Timestamp('2016-03-29 08:22:00')), :]

df_filled_full_lines.loc[(1573, slice(None), 1), ['line_value', 'line_increment']]
#df_filled_full_lines.loc[(1573, slice(None), 1), ['line_value', 'line_increment']].to_csv(r'c:\work\others\Oleg\test.csv', sep=';', decimal=',')

df_full_lines.loc[(1573, 1, 1), 'line_value']
df_filled_full_lines.loc[(1573, 1, 1), 'line_value']
df_filled_full_lines.loc[1573, ('line_value', 'line_increment')].to_csv(r'c:\work\others\Oleg\test.csv', sep=';', decimal=',')

#full_index = pd.MultiIndex.from_product([df_lines.groupby(['match_ref', 'snapshot_time']).groups.keys(), all_houses, all_rtv_refs], names=['match_ref', 'snapshot_time', 'house_ref', 'RTV_ref'])

#РИСУЕМ ГРАФИКИ 
#%pyplot
#fig = plt.figure()
#axis = fig.add_subplot(111)

#uniq_line_values = df_filled_full_lines['line_value'].groupby(level=[0, 1, 2]).apply(lambda x: len(x.unique()))
uniq_line_values = df_filled_full_lines['line_value'].groupby(level=[0, 1, 2]).nunique()

longest_uniq_line_values = 1/df_filled_full_lines.loc[uniq_line_values.idxmax(), 'line_value']
longest_uniq_line_values.plot()

plt.show()

ser_std = df_filled_full_lines['line_value'].groupby(level=[0, 1, 2]).std()

variest_line_values = 1/df_filled_full_lines.loc[ser_std.idxmax()]['line_value']
variest_line_values.plot()
plt.show()

почему loc удаляет верхние уровни из MultiIndex?

