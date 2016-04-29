Операционная таблица. Все изменения ставок по всем конторам регистрируются здесь. 
TS_ref - ссылка на global time snapshot - временную метку начала процессинга ставок по определеной группе турниров.
line-value и line_increment - величина и инкремент ставки
snapshot_time - время контроля ставки 
is_it_starting - признак начальной ставки: ставки-открытия линии конторы
RTV_ref - вид ставки (1, Х или 2: победа первой команды, ничья, победа второй команды)
time_increment - приращение времени с последнего контроля ставки

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

df = pd.read_sql('select * from Houses', conn)


/****** Script for SelectTopNRows command from SSMS  ******/
SELECT TOP 1000 [line_ID]
      ,[house_ref]
      ,[match_ref]
      ,[TS_ref]
      ,[line_value]
      ,[line_increment]
      ,[snapshot_time]
      ,[is_it_starting]
      ,[RTV_ref]
      ,[time_increment]
  FROM [BRefDB].[dbo].[Lines]
  where TS_ref = 1878
  order by is_it_starting, snapshot_time, line_ID
  ;

df_lines = pd.read_sql('select * from Lines l where TS_ref = 1878 and snapshot_time = CONVERT(DATETIME, \'2016-03-13 10:39:00.000\') and RTV_Ref = 1 order by is_it_starting, snapshot_time, line_ID', conn)

df_features = df_lines.pivot(index='match_ref', columns='house_ref', values='line_value')

df_lines = pd.read_sql('select * from Lines l where TS_ref = 1878 and RTV_Ref = 1 order by is_it_starting, snapshot_time, line_ID', conn)

df_features = df_lines.pivot(index=['match_ref', 'snapshot_time'], columns='house_ref', values='line_value')

table = pd.pivot_table(df_lines, index=['match_ref', 'snapshot_time'], columns=['house_ref'], values='line_value')

select *
from Lines l 
where 1=1
and match_ref = 1754 
and TS_ref = 1881
and RTV_Ref = 1 
order by house_ref, line_ID
;

select l.match_ref, count(*) 
from Lines l
group by l.match_ref
order by count(*) desc
;

select match_ref, l.house_ref
  , MAX(CASE WHEN l.is_it_starting = 1 THEN l.line_value END) start_value
  , MAX(CASE WHEN l.is_it_starting = 0 THEN l.line_value END) next_value
  , MAX(CASE WHEN l.is_it_starting = 0 THEN l.line_increment END) line_increment
from Lines l 
where 1=1
--and match_ref = 1754 
and TS_ref = 1881
and RTV_Ref = 1 
GROUP BY match_ref, l.house_ref
order by match_ref, l.house_ref
;

df_lines = pd.read_sql('''select match_ref, l.house_ref
  , MAX(CASE WHEN l.is_it_starting = 1 THEN l.line_value END) start_value
  , MAX(CASE WHEN l.is_it_starting = 0 THEN l.line_value END) next_value
  --, MAX(CASE WHEN l.is_it_starting = 0 THEN l.line_increment END) line_increment
from Lines l 
where 1=1
--and match_ref = 1754 
and TS_ref = 1881
and RTV_Ref = 1 
GROUP BY match_ref, l.house_ref
order by match_ref, l.house_ref''', conn)

df_features = pd.pivot_table(df_lines, index=['match_ref'], columns=['house_ref'], values=['start_value', 'next_value', 'line_increment'])

df_match_results = pd.read_sql('SELECT match_ref, CASE WHEN RTV_ref = 1 THEN 1 ELSE 0 END result FROM Match_results', conn, index_col='match_ref')
df_results = df_features.join(df_match_results, how='inner')

#features = df_results.drop('result', axis=1).fillna(value=np.finfo(np.float32).min + 1).values
#features = df_results.drop('result', axis=1).fillna(method='ffill').fillna(method='bfill').values
features = df_results.drop('result', axis=1).fillna(value=0).values
scaled_features = scale(features, axis=0)
target = df_results['result'].values



kf = KFold(scaled_features.shape[0], n_folds=5, shuffle=True, random_state=42)

Cs = [10**x for x in range(-5, 6)]

lr = LogisticRegression()

clf = grid_search.GridSearchCV(estimator=lr, param_grid=dict(C=Cs), n_jobs=4, cv=kf, scoring='roc_auc')
clf.fit(scaled_features, target)


df_lines_test = pd.read_sql('''select match_ref, l.house_ref
  , MAX(CASE WHEN l.is_it_starting = 1 THEN l.line_value END) start_value
  , MAX(CASE WHEN l.is_it_starting = 0 THEN l.line_value END) next_value
  --, MAX(CASE WHEN l.is_it_starting = 0 THEN l.line_increment END) line_increment
from Lines l 
where 1=1
--and match_ref = 1754 
and TS_ref = 1880
and RTV_Ref = 1 
GROUP BY match_ref, l.house_ref
order by match_ref, l.house_ref''', conn)

df_features_test = pd.pivot_table(df_lines_test, index=['match_ref'], columns=['house_ref'], values=['start_value', 'next_value', 'line_increment'])
#df_results_test = df_features_test.join(df_match_results, how='inner')
#features_test = df_results_test.drop('result', axis=1).fillna(value=0).values
features_test = df_features_test.fillna(value=0).values
scaled_features_test = scale(features_test, axis=0)

predict = clf.predict_proba(scaled_features_test)

#### попытка номер два



df_lines = pd.read_sql('''select match_ref, CAST(TS_ref as VARCHAR) + CAST(is_it_starting AS VARCHAR) full_ts_ref, l.house_ref--, l.is_it_starting
                            , MAX(CASE WHEN l.RTV_Ref = 1 THEN l.line_value END) first_win
                            , MAX(CASE WHEN l.RTV_Ref = 2 THEN l.line_value END) no_win
                            , MAX(CASE WHEN l.RTV_Ref = 3 THEN l.line_value END) second_win
                          from Lines l 
                          where 1=1
                          --and match_ref = 1754 
                          --and TS_ref = 1881
                          --and RTV_Ref = 1 
                          GROUP BY match_ref, CAST(TS_ref as VARCHAR) + CAST(is_it_starting AS VARCHAR), l.house_ref--, l.is_it_starting
                          ''', conn, index_col='[match_ref, full_ts_ref, house_ref]')



df_lines = pd.read_sql('''select * from Lines l where is_it_starting = 0 and RTV_ref in (1, 2, 3)''', conn)
df_lines.set_index('line_ID', inplace=True)

df_match_results = pd.read_sql('SELECT match_ref, CASE WHEN RTV_ref = 1 THEN 1 ELSE 0 END result FROM Match_results', conn, index_col='match_ref')
df_lines_n_results = df_lines.join(df_match_results, on='match_ref', how='inner')

df_features = df_lines_n_results.pivot_table(index=['match_ref', 'TS_ref', 'house_ref'], columns=['RTV_ref'], values=['line_value', 'line_increment', 'result'])

df_filled_features = df_features.fillna(method='ffill')


#features = df_results.drop('result', axis=1).fillna(value=np.finfo(np.float32).min + 1).values
#features = df_results.drop('result', axis=1).fillna(method='ffill').fillna(method='bfill').values
#features = df_results.drop('result', axis=1).fillna(value=0).values
#features = df_filled_features.drop('result', axis=1).values
features = df_filled_features.drop('line_increment', axis=1).drop('result', axis=1).values
scaled_features = scale(features, axis=0)
target = df_filled_features['result', 1].values



kf = KFold(scaled_features.shape[0], n_folds=5, shuffle=True, random_state=42)

Cs = [10**x for x in range(-5, 6)]

lr = LogisticRegression()

clf = grid_search.GridSearchCV(estimator=lr, param_grid=dict(C=Cs), n_jobs=4, cv=kf, scoring='roc_auc')
clf.fit(scaled_features, target)
clf.best_estimator_.coef_

## проверка значений 

SELECT l.RTV_ref prognosis, mr.RTV_ref result, 1/l.line_value prob, l.* 
FROM Lines l 
  INNER JOIN (SELECT match_ref, MAX(TS_ref) last_TS_ref from Lines GROUP BY match_ref) last_ts
  ON l.match_ref = last_ts.match_ref and l.TS_ref = last_ts.last_TS_ref
 INNER JOIN Match_results mr ON l.match_ref = mr.match_ref
WHERE l.RTV_ref < 4
 ORDER BY ABS(l.RTV_ref - mr.RTV_ref), 1/l.line_value
 ;

SELECT COUNT(*), CAST(l.RTV_ref as VARCHAR) + CAST(mr.RTV_ref as VARCHAR), sign(1/l.line_value - .5) ok --,  l.RTV_ref prognosis, mr.RTV_ref result,1/l.line_value prob, l.* 
FROM Lines l 
  INNER JOIN (SELECT match_ref, MAX(TS_ref) last_TS_ref from Lines GROUP BY match_ref) last_ts
  ON l.match_ref = last_ts.match_ref and l.TS_ref = last_ts.last_TS_ref
 INNER JOIN Match_results mr ON l.match_ref = mr.match_ref
WHERE l.RTV_ref < 4
 GROUP BY CAST(l.RTV_ref as VARCHAR) + CAST(mr.RTV_ref as VARCHAR), sign(1/l.line_value - .5)
 ORDER BY CAST(l.RTV_ref as VARCHAR) + CAST(mr.RTV_ref as VARCHAR), sign(1/l.line_value - .5)
 ;


df_547 = df_lines[df_lines.match_ref == 547]
df_lines.sort('TS_ref', ascending=False).groupby(['match_ref', 'house_ref', 'RTV_ref'], as_index=False).first()
df_547.sort_values(by='TS_ref', ascending=False).groupby(['match_ref', 'house_ref', 'RTV_ref'], as_index=False).first()

-- разреженная матрица всех прогнозов для значений (1, 2, 3) для матча 547
SELECT all_dims.*, l.line_value, l.line_increment, l.snapshot_time, l.time_increment 
FROM (SELECT * FROM Lines WHERE RTV_ref in (1, 2, 3) AND match_ref = 547 AND is_it_starting = 0) l 
  RIGHT OUTER JOIN
(
SELECT * from
  (SELECT DISTINCT TS_ref from Lines WHERE match_ref = 547 AND RTV_ref in (1, 2, 3)) all_ts,
  (SELECT DISTINCT house_ref from Lines WHERE match_ref = 547 AND RTV_ref in (1, 2, 3)) all_houses,
  (SELECT 1 RTV_ref UNION ALL SELECT 2 UNION ALL SELECT 3) all_rtvs
  ) all_dims
  ON all_dims.TS_ref = l.TS_ref and all_dims.house_ref = l.house_ref and all_dims.RTV_ref = l.RTV_ref
ORDER BY all_dims.TS_ref, all_dims.house_ref, all_dims.RTV_ref
;


sql_bets_format = '''SELECT all_dims.TS_ref, all_dims.house_ref, all_dims.RTV_ref, 1/l.line_value prob_value--, 1/l.line_increment, l.snapshot_time, l.time_increment 
FROM (SELECT * FROM Lines WHERE RTV_ref in (1, 2, 3) AND match_ref = {0} AND is_it_starting = 0) l 
  RIGHT OUTER JOIN
(
SELECT * from
  (SELECT DISTINCT TS_ref from Lines WHERE match_ref = {0} AND RTV_ref in (1, 2, 3) AND is_it_starting = 0) all_ts,
  (SELECT DISTINCT house_ref from Lines WHERE match_ref = {0} AND RTV_ref in (1, 2, 3)) all_houses,
  (SELECT 1 RTV_ref UNION ALL SELECT 2 UNION ALL SELECT 3) all_rtvs
  ) all_dims
  ON all_dims.TS_ref = l.TS_ref and all_dims.house_ref = l.house_ref and all_dims.RTV_ref = l.RTV_ref
ORDER BY all_dims.TS_ref, all_dims.house_ref, all_dims.RTV_ref'''

sql_bets = sql_bets_format.format(2698)

df_bets = pd.read_sql(sql_bets, conn)

df_bets_by_houses = df_bets.pivot_table(index=['TS_ref'], columns=['house_ref', 'RTV_ref'], values=['prob_value'])
df_filled_bets_by_houses = df_bets_by_houses.fillna(method='ffill')

fig = plt.figure()
axis = fig.add_subplot(111)

%matplotlib
df_first_line = df_filled_bets_by_houses['prob_value', 1]

df_first_line.plot()



sql_prob = '''SELECT l.RTV_ref prognosis, mr.RTV_ref result, 1/l.line_value prob_value, l.house_ref, l.match_ref--, m.match_name 
                    FROM Lines l 
                      INNER JOIN (SELECT match_ref, MAX(TS_ref) last_TS_ref from Lines WHERE is_it_starting = 0 GROUP BY match_ref) last_ts
                      ON l.match_ref = last_ts.match_ref and l.TS_ref = last_ts.last_TS_ref
                     INNER JOIN Match_results mr ON l.match_ref = mr.match_ref
                     INNER JOIN Matches m ON l.match_ref = m.match_ID
                    WHERE l.RTV_ref < 4 AND l.is_it_starting = 0'''

#df_prob = pd.read_sql(sql_prob, conn, index_col=['match_ref', 'house_ref'])
df_prob = pd.read_sql(sql_prob, conn)

df_prob_by_result = df_prob.pivot_table(index=['match_ref', 'house_ref'], columns=['prognosis'], values=['prob_value', 'result'])
