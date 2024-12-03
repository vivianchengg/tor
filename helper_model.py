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

# get final recommendation
def genRanking():
  df = pd.read_csv(csvName)
  df['weight'] = df.groupby('cusip', group_keys=False).apply(getWeight)
  weighted_ranking = (
    df.groupby('cusip')
    .apply(lambda group: (group['ranking_target'] * group['weight']).sum() / group['weight'].sum())
    .reset_index(name='weighted_ranking')
    .sort_values('weighted_ranking', ascending=False)
  )

  # add issuer name
  issuers = df[['cusip', 'nameOfIssuer']].drop_duplicates()
  weighted_ranking = pd.merge(weighted_ranking, issuers, on='cusip', how='left')

  rec_no = 10
  top_recommendations = weighted_ranking.head(rec_no)
  print(f"Final Stock Recommendations (Top {rec_no} Ranked):")
  for _, row in top_recommendations.iterrows():
    print(f"CUSIP: {row['cusip']}, Issuer: {row['nameOfIssuer']}, Weighted Ranking Score: {row['weighted_ranking']:.2f}")

def getWeight(group):
  periods = group['periodOfReport'].sort_values(ascending=False).unique()
  weights = {period: 1 - (i / len(periods)) for i, period in enumerate(periods)}
  weight_map = group['periodOfReport'].map(weights)
  return weight_map

def runModeling():
  # modeling()
  genRanking()