#! /usr/bin/env python3

import pandas as pd

csvName = 'berkshire_holdings_combined.csv'

def sortData(df):
  df['periodOfReport'] = pd.to_datetime(df['periodOfReport'])
  df = df.sort_values(by=['cusip', 'periodOfReport'], ascending=[True, True]).reset_index(drop=True)
  df['value_change_pct'] = df.groupby('cusip')['value'].pct_change()
  df['shares_change_pct'] = df.groupby('cusip')['shrs_or_prn_amt'].pct_change()
  return df

def saveData(allData):
  allData = sortData(allData)
  allData.to_csv(csvName, index=False)
  print(f"Data saved to '{csvName}'.")
  return allData