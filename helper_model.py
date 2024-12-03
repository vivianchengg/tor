#! /usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import ndcg_score

csvName = 'berkshire_holdings_combined.csv'

def modeling():
  df = pd.read_csv(csvName)

  features = ['ROE', 'D/E Ratio', 'profit_margin', 'P/B ratio', 'beta', 'P/E ratio', 'dividend_score']
  x = df[features]
  y = df['ranking_target']

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

  # array of group numbers per report period
  train_groups = df.loc[x_train.index, 'periodOfReport'].value_counts(sort=False).values
  test_groups = df.loc[x_test.index, 'periodOfReport'].value_counts(sort=False).values

  train_data = lgb.Dataset(x_train, label=y_train, group=train_groups)
  test_data = lgb.Dataset(x_test, label=y_test, group=test_groups, reference=train_data)

  max_rank = max(df['ranking_target'].unique())

  params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_at': [3, 5, 10],
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'label_gain': list(range(max_rank + 1))
  }

  model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, test_data],
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
  )
  y_pred = model.predict(x_test)

  ndcg = ndcg_score([y_test], [y_pred], k=5)
  print(f"NDCG@5: {ndcg}")

  importance = model.feature_importance(importance_type='gain')
  features = x_train.columns
  feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=False)
  print(feature_importance)

def runModeling():
  modeling()