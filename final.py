import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import grid_search
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.preprocessing import scale, StandardScaler
from sklearn.metrics import roc_auc_score, make_scorer
import time
import datetime


# Loading
features = pd.read_csv('final/features.csv', index_col='match_id')
#features.head()

features_test = pd.read_csv('final/features_test.csv', index_col='match_id')

# Preprocessing 
not_features_columns = set(features.columns) - set(features_test.columns)
not_features_columns.add('start_time')

clean_features = features.drop(not_features_columns, axis=1)
count_clean_features = clean_features.count()
count_missing = count_clean_features[count_clean_features < 97230]

filled_clean_features = clean_features.fillna(value=np.finfo(np.float32).min + 1)
target = features['radiant_win']

# Approach 1
kf = KFold(filled_clean_features.shape[0], n_folds=5, shuffle=True, random_state=42)

gbc = GradientBoostingClassifier(verbose=True, random_state=241)
tree_cases = [n_trees for n_trees in range(10, 50, 10)]

roc_auc_scorer = make_scorer(roc_auc_score, needs_threshold=True)

#clf = grid_search.GridSearchCV(estimator=gbc, param_grid=dict(n_estimators=tree_cases), n_jobs=4, cv=kf, scoring=roc_auc_scorer)
#clf.fit(filled_clean_features.as_matrix(), target.as_matrix())
#clf.best_params_
#clf.best_score_
#clf.grid_scores_
scores = []
for n_tree in tree_cases:
  start_time = datetime.datetime.now()

  kf = KFold(non_categorial_scaled_features.shape[0], n_folds=5, shuffle=True, random_state=42)
  gbc = GradientBoostingClassifier(verbose=True, random_state=241)
  scores.append((n_tree, np.mean(cross_val_score(gbc, filled_clean_features.as_matrix(), target.as_matrix(), cv=kf, scoring=roc_auc_scorer))))

  print 'Time elapsed:', datetime.datetime.now() - start_time

# half population with max_depth=2
import random
choice = random.sample(filled_clean_features.index, filled_clean_features.shape[0] / 2)
choiced_filled_clean_features = filled_clean_features.ix[choice]
choiced_target = features.ix[choice]['radiant_win']
choiced_kf = KFold(choiced_filled_clean_features.shape[0], n_folds=5, shuffle=True, random_state=42)

choiced_gbc = GradientBoostingClassifier(verbose=True, random_state=241, max_depth=2)
tree_cases = [n_trees for n_trees in range(10, 50, 10)]

#choiced_clf = grid_search.GridSearchCV(estimator=choiced_gbc, param_grid=dict(n_estimators=tree_cases), n_jobs=4, cv=choiced_kf, scoring=roc_auc_scorer)
#choiced_clf.fit(choiced_filled_clean_features.as_matrix(), choiced_target.as_matrix())
scores = []
for n_tree in tree_cases:
  start_time = datetime.datetime.now()

  kf = KFold(non_categorial_scaled_features.shape[0], n_folds=5, shuffle=True, random_state=42)
  choiced_gbc = GradientBoostingClassifier(verbose=True, random_state=241, max_depth=2)
  scores.append((n_tree, np.mean(cross_val_score(choiced_gbc, choiced_filled_clean_features.as_matrix(), choiced_target.as_matrix(), cv=kf, scoring=roc_auc_scorer))))

  print 'Time elapsed:', datetime.datetime.now() - start_time

# Approach 2
from sklearn.linear_model import LogisticRegression

zeroed_clean_features = clean_features.fillna(value=0)
scaled_features = scale(zeroed_clean_features, axis=0)

lr_kf = KFold(scaled_features.shape[0], n_folds=5, shuffle=True, random_state=42)

Cs = [10**x for x in range(-5, 6)]

lr = LogisticRegression()

#lr_clf = grid_search.GridSearchCV(estimator=lr, param_grid=dict(C=Cs), n_jobs=4, cv=lr_kf, scoring=roc_auc_scorer)
#lr_clf.fit(scaled_features, target.as_matrix())

scores = []
for c in Cs:
  start_time = datetime.datetime.now()

  kf = KFold(non_categorial_scaled_features.shape[0], n_folds=5, shuffle=True, random_state=42)
  #rfr = RandomForestRegressor(random_state=1, n_estimators=n_tree)
  lr = LogisticRegression(C=c)
  #scores.append((n_tree, np.mean(cross_val_score(rfr, features, target, cv=kf, scoring=make_scorer(r2_score)))))
  scores.append((c, np.mean(cross_val_score(lr, non_categorial_scaled_features, target, cv=kf, scoring='roc_auc'))))

  print 'Time elapsed:', datetime.datetime.now() - start_time

# without categorical features
#categorial_feature_names = ['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']
heroes_columns = [gamer + str(num) + '_hero' for gamer in ('r', 'd') for num in xrange(1, 6)]
categorial_feature_names = ['lobby_type'] + heroes_columns

non_categorial_scaled_features = scale(zeroed_clean_features.drop(categorial_feature_names, axis=1), axis=1)

nc_lr_kf = KFold(non_categorial_scaled_features.shape[0], n_folds=5, shuffle=True, random_state=42)
nc_lr_clf = grid_search.GridSearchCV(estimator=lr, param_grid=dict(C=Cs), n_jobs=4, cv=nc_lr_kf, scoring='roc_auc')
nc_lr_clf.fit(non_categorial_scaled_features, target.as_matrix())

scores = []
for c in Cs:
  start_time = datetime.datetime.now()

  kf = KFold(non_categorial_scaled_features.shape[0], n_folds=5, shuffle=True, random_state=42)
  #rfr = RandomForestRegressor(random_state=1, n_estimators=n_tree)
  lr = LogisticRegression(C=c)
  #scores.append((n_tree, np.mean(cross_val_score(rfr, features, target, cv=kf, scoring=make_scorer(r2_score)))))
  scores.append((c, np.mean(cross_val_score(lr, non_categorial_scaled_features, target, cv=kf, scoring=roc_auc_scorer))))

  print 'Time elapsed:', datetime.datetime.now() - start_time

#max(enumerate(scores), key=lambda x: x[1])
max(scores, key=lambda x: x[1])


  start_time = datetime.datetime.now()
  roc_auc_scorer = make_scorer(roc_auc_score)
  kf = KFold(non_categorial_scaled_features.shape[0], n_folds=5, shuffle=True, random_state=42)
  #rfr = RandomForestRegressor(random_state=1, n_estimators=n_tree)
  lr = linear_model.LogisticRegression(C=1e-04, n_jobs=4, verbose=1)
  #scores.append((n_tree, np.mean(cross_val_score(rfr, features, target, cv=kf, scoring=make_scorer(r2_score)))))
  scores.append((c, np.mean(cross_val_score(lr, non_categorial_scaled_features, target.as_matrix(), cv=kf, scoring=roc_auc_scorer))))
  print 'Time elapsed:', datetime.datetime.now() - start_time


  features[heroes_columns]

  heroes_count = features[heroes_columns].apply(pd.value_counts)
  heroes_test_count = features_test[heroes_columns].apply(pd.value_counts)

  all_heroes  = {id:i for i, id in enumerate(sorted(list(set(heroes_count.index.values) | set(heroes_test_count.index.values))))}

  #heroes_set = set(heroes_count.index.values)
  #heroes_test_set = set(heroes_count_test.index.values)
  
  bag_of_heroes = np.zeros((non_categorial_scaled_features.shape[0], len(all_heroes)))

  for i, match_id in enumerate(features.index):
    for p in range(1, 6):
        bag_of_heroes[i, all_heroes[features.ix[match_id, 'r%d_hero' % p]]] = 1
        bag_of_heroes[i, all_heroes[features.ix[match_id, 'd%d_hero' % p]]] = -1

  scaled_features_with_bag_of_heroes = np.hstack((non_categorial_scaled_features, bag_of_heroes))

scoresBOH = []
for c in Cs:
  start_time = datetime.datetime.now()

  kf = KFold(scaled_features_with_bag_of_heroes.shape[0], n_folds=5, shuffle=True, random_state=42)
  #rfr = RandomForestRegressor(random_state=1, n_estimators=n_tree)
  lr = LogisticRegression(C=c)
  #scores.append((n_tree, np.mean(cross_val_score(rfr, features, target, cv=kf, scoring=make_scorer(r2_score)))))
  scoresBOH.append((c, np.mean(cross_val_score(lr, scaled_features_with_bag_of_heroes, target, cv=kf, scoring='roc_auc'))))

  print ('Time elapsed:', datetime.datetime.now() - start_time)
