#! /usr/bin/env python3

import time
import pandas as pd
import requests
import yfinance as yf

csvName = 'berkshire_holdings_combined.csv'
API_KEY = '4c601dea-c4c1-4f51-8fda-1175163c35b6'

updated_ticker_mapping = {
  'ATVI': 'AIY.DE',
  'LENB': 'LEN-B'
}

outdatedCusips = ['00507V109', '862121100']

def cusip_to_ticker_batch(cusips, dict):
  headers = {
      'Content-Type': 'application/json',
      'X-OPENFIGI-APIKEY': API_KEY
  }

  max_batch_size = 100  # OpenFIGI allows up to 100 jobs per request

  for i in range(0, len(cusips), max_batch_size):
    batch = cusips[i:i + max_batch_size]
    data = [{"idType": "ID_CUSIP", "idValue": cusip} for cusip in batch]

    response = requests.post('https://api.openfigi.com/v3/mapping', headers=headers, json=data)

    if response.status_code == 200:
      batch_results = response.json()
      for result, cusip in zip(batch_results, batch):
        if 'data' in result and result['data']:
          ticker = result['data'][0].get('ticker', None)
        else:
          ticker = None
        dict[cusip] = ticker
    else:
      print(f"Error {response.status_code}: {response.text}")

    time.sleep(0.25)  # max 25 requests per 6 seconds

  return dict

def fetchYF(ticker):
  try:
    stock = yf.Ticker(ticker)

    financials = stock.financials
    net_income = financials.loc['Net Income'] if 'Net Income' in financials.index else None

    total_debt = stock.info.get('totalDebt', None)

    balance_sheet = stock.balancesheet
    stockholders_equity = balance_sheet.loc['Stockholders Equity'] if 'Stockholders Equity' in balance_sheet.index else None

    roe = net_income / stockholders_equity if net_income is not None and stockholders_equity is not None else None
    de_ratio = total_debt / stockholders_equity if total_debt is not None and stockholders_equity is not None else None

    print(f"net-income: {net_income}")
    print(f"debt: {total_debt}")
    print(f"equity: {stockholders_equity}")
    print(f"fetching: roe->{roe}, de_ratio->{de_ratio}")
    return roe, de_ratio
  except Exception as e:
    print(f"Error fetching data for {ticker}: {e}")
    return None, None

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

def handleROEDECols(df):
  df['ROE'] = None
  df['D/E Ratio'] = None

  for index, row in df.iterrows():
    ticker = row['ticker']
    ticker = updated_ticker_mapping.get(ticker, ticker)
    period = row['periodOfReport']
    year = int(period.split('-')[0].strip())

    # if ticker != 'STOR':
    #   continue

    print('--------')
    print(ticker)

    roe, de_ratio = fetchYF(ticker)

    if roe is not None:
      try:
        roe_value = roe.loc[roe.index.year == year].iloc[0]
        df.at[index, 'ROE'] = roe_value
        print(f"roe: {roe_value}")

      except IndexError:
        print(f"No ROE data for year: {year} for {ticker}")
    else:
      print('ROE is None')

    if de_ratio is not None:
      try:
        de_value = de_ratio.loc[de_ratio.index.year == year].iloc[0]
        df.at[index, 'D/E Ratio'] = de_value
        print(f"de: {de_value}")

      except IndexError:
        print(f"No D/E ratio data for year: {year} for {ticker}")
    else:
      print('de is none')

    print('--------')

  return df

def handleFundColumns(df):
  tmpCusip_df = df[['cusip', 'nameOfIssuer']].drop_duplicates()
  ticker_df = pd.read_csv('tickerData.csv')
  cusip_mapping = dict(zip(ticker_df['CUSIP'], ticker_df['SYMBOL']))
  issuer_mapping = dict(zip(ticker_df['DESCRIPTION'], ticker_df['SYMBOL']))
  mapping_dict = {}

  for _, tmp_row in tmpCusip_df.iterrows():
    tmp_cusip = tmp_row['cusip']
    tmp_issuer = tmp_row['nameOfIssuer'].strip()

    if tmp_cusip in cusip_mapping:
      mapping_dict[tmp_cusip] = cusip_mapping[tmp_cusip]
    elif tmp_issuer in issuer_mapping:
      mapping_dict[tmp_cusip] = issuer_mapping[tmp_issuer]

  unmatched = [c for c in tmpCusip_df['cusip'].unique() if c not in mapping_dict]
  mapping_dict = cusip_to_ticker_batch(unmatched, mapping_dict)

  df['ticker'] = df['cusip'].map(mapping_dict)

  df = handleROEDECols(df)

  df.to_csv(csvName, index=False)

def handleOutdatedComp(df):
  df = df[~df['cusip'].isin(outdatedCusips)]
  return df

def analyse():
  df = pd.read_csv(csvName)
  df = df.drop_duplicates()

  df = handleOutdatedComp(df)
  handleFundColumns(df)

  for cusip in df['cusip'].unique():
    cData = df[df['cusip'] == cusip]
    checkStock(cData)
    break