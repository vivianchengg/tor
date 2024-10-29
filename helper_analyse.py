#! /usr/bin/env python3

import pandas as pd


csvName = 'berkshire_holdings_combined.csv'

def checkStock(df):
  prevTime = None
  prevVal = -1
  prevAmt = -1
  for period in df['periodOfReport'].unique():
    sData = df[df['periodOfReport'] == period]
    company = sData['nameOfIssuer'].values[0]
    cusip = sData['cusip'].values[0]
    if prevVal == -1 and prevAmt == -1 and prevTime is None:
      prevVal, prevAmt, prevTime = sData['value'].values[0], sData['shrs_or_prn_amt'].values[0], sData['periodOfReport'].values[0]
      continue
    else:
      curVal, curAmt, curTime = sData['value'].values[0], sData['shrs_or_prn_amt'].values[0], sData['periodOfReport'].values[0]

      print(f"from {prevTime} to {curTime}: {company}({cusip}) - {prevAmt}/{prevVal} -> {curAmt}/{curVal}")
      prevVal, prevAmt, prevTime = curVal, curAmt, curTime

def analyse():
  df = pd.read_csv(csvName)
  df = df.drop_duplicates()
  for cusip in df['cusip'].unique():
    cData = df[df['cusip'] == cusip]
    checkStock(cData)
    break