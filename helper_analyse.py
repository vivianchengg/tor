#! /usr/bin/env python3

import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import json

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

def fetchROEDE(ticker):
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

    roe, de_ratio = fetchROEDE(ticker)

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

  return df

def handleOutdatedComp(df):
  df = df[~df['cusip'].isin(outdatedCusips)]
  return df

def handleMissingVal(df):
  df['ROE'] = df.groupby('cusip')['ROE'].transform(lambda x: x.ffill())
  df['ROE'] = df.groupby('cusip')['ROE'].transform(lambda x: x.bfill())
  avgDE = df.groupby('cusip')['D/E Ratio'].transform('mean')
  df['D/E Ratio'] = df['D/E Ratio'].fillna(avgDE)
  df = df.dropna(subset=['industry'])
  df['P/E ratio'] = df.groupby('cusip')['P/E ratio'].transform(lambda x: x.ffill())
  return df

def fetch_industry(ticker):
  try:
    stock = yf.Ticker(ticker)
    industry = stock.info.get('industry', None)
    return industry
  except Exception as e:
    print(f"Error fetching industry for {ticker}: {e}")
    return None
    
def fetch_profit_margin(ticker):
  try:
    stock = yf.Ticker(ticker)
    financials = stock.financials
    net_income = financials.loc['Net Income'] if 'Net Income' in financials.index else None
    revenue = financials.loc['Total Revenue'] if 'Total Revenue' in financials.index else None
    profit_margin = net_income / revenue if net_income is not None and revenue else None
    return profit_margin
  except Exception as e:
    print(f"Error for {ticker}: {e}")
    return None

def fetch_pb_ratio(ticker):
  try:
    stock = yf.Ticker(ticker)
    price = stock.info.get('currentPrice', None)
    balance_sheet = stock.balancesheet
    total_equity = balance_sheet.loc['Stockholders Equity'] if 'Stockholders Equity' in balance_sheet.index else None
    outstanding_shares = stock.info.get('sharesOutstanding', None)
    bvps = total_equity / outstanding_shares if total_equity is not None and outstanding_shares is not None else None
    pb_ratio = price / bvps if price is not None and bvps is not None else None
    return pb_ratio
  except Exception as e:
    print(f"Error fetching P/B Ratio for {ticker}: {e}")
    return None

def fetch_pe_ratio(ticker, period):
  try:
    stock = yf.Ticker(ticker)

    end_date = pd.to_datetime(period)
    start_date = end_date - pd.DateOffset(months=3)
    year = int(period.split('-')[0].strip())

    # closing price of the quarter
    price = stock.history(period='3mo', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))['Close'].mean()

    # eps
    earnings_data = stock.financials.loc['Net Income'] if 'Net Income' in stock.financials.index else None
    shares_outstanding = stock.info.get('sharesOutstanding', None)
    eps = (earnings_data / shares_outstanding).loc[earnings_data.index.year == year].iloc[0] if earnings_data is not None and shares_outstanding else None

    # P/E ratio
    pe_ratio = price / eps if price is not None and eps is not None else None
    print(f"P/E Ratio for {ticker} in {year}: {pe_ratio}")
    return pe_ratio
  except Exception as e:
    print(f"Error fetching P/E Ratio for {ticker} in {year}: {e}")
    return None
  
def fetch_dividend_score(ticker, period):
  try:
    stock = yf.Ticker(ticker)

    end_date = pd.to_datetime(period)
    start_date = end_date - pd.DateOffset(months=3)
    year = int(period.split('-')[0].strip())

    dividends = stock.dividends
    dividends.index = dividends.index.tz_convert(None)
    div = dividends.loc[start_date:end_date].sum()
    price = stock.history(period='3mo', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))['Close'].mean()

    # payout ratio
    financials = stock.financials
    netI = financials.loc['Net Income'] if 'Net Income' in financials.index else None
    net_income = netI.loc[netI.index.year == year].iloc[0]
    shares_outstanding = stock.info.get('sharesOutstanding', None)
    eps = (net_income / shares_outstanding) if net_income is not None and shares_outstanding else None
    dps = div / shares_outstanding if div is not None and shares_outstanding else None
    payout_ratio = dps / eps if dps is not None and eps else None
    adjustedPR = max(0, 1 - min(payout_ratio, 1)) if payout_ratio is not None else 0

    # dividend yield
    dYield = div / price if div is not None and price else 0

    # dividend growth
    previous_dividends = dividends.loc[start_date - pd.DateOffset(months=3):start_date].sum()
    dGrowth = ((div - previous_dividends) / previous_dividends) if div and previous_dividends else 0

    combined_score = 0.5 * adjustedPR + 0.3 * dGrowth + 0.2 * dYield
    print(f"combined score for {ticker}: {combined_score}")
    return combined_score
  except Exception as e:
    print(f"Error fetching dividend score for {ticker}: {e}")
    return None

def fetch_beta(ticker):
  try:
    stock = yf.Ticker(ticker)
    beta = stock.info.get('beta', None)
    return beta
  except Exception as e:
    print(f"Error fetching beta for {ticker}: {e}")
    return None

def handleColWYr(df, cols):
  for col in cols:
    df[col] = None
    for index, row in df.iterrows():
      ticker = row['ticker']
      period = row['periodOfReport']
      year = int(period.split('-')[0].strip())

      target = None
      if col == 'profit_margin':
        target = fetch_profit_margin(ticker)
      elif col == 'P/B ratio':
        target = fetch_pb_ratio(ticker)
      elif col == 'dividend_score':
        target = fetch_dividend_score(ticker, period)
      elif col == 'P/E ratio':
        target = fetch_pe_ratio(ticker, period)

      if target is not None:
        try:
          target_value = None
          if col == 'P/E ratio' or col == 'dividend_score':
            target_value = target
          else:
            target_value = target.loc[target.index.year == year].iloc[0]
          df.at[index, col] = target_value
          print(f"{col} for year {year} - {ticker}: {target_value}")
        except IndexError:
          print(f"No {col} data for year: {year} for {ticker}")
      else:
        print(f'{col} ({ticker}) is None')

  return df

def handleMoreColumns(df):
  df['industry'] = df['ticker'].apply(fetch_industry)
  df = handleColWYr(df, ['profit_margin', 'P/B ratio', 'dividend_score'])
  df['Industry_Avg_ROE'] = df.groupby('industry')['ROE'].transform('mean')
  df['Industry_Avg_DE'] = df.groupby('industry')['D/E Ratio'].transform('mean')
  df['beta'] = df['ticker'].apply(fetch_beta)
  df['P/E ratio'] = df['ticker'].apply(fetch_pe_ratio)
  return df

def cleanData(df):
  # ROE
  df = df.dropna(subset=['ROE'])

  # profit margin, dividend score
  df['profit_margin'] = df['profit_margin'].fillna(0)
  df['dividend_score'] = df['dividend_score'].fillna(0.5)

  # beta
  df['beta'].fillna(1, inplace=True)

  # P/B ratio, P/E ratio
  df['P/B ratio'] = df.groupby('cusip')['P/B ratio'].transform(lambda x: x.fillna(x.mean()))
  df['P/E ratio'] = df.groupby('cusip')['P/E ratio'].transform(lambda x: x.fillna(x.mean()))

  return df

def scaleData(df):
  scaler = StandardScaler()
  columns = ['ROE', 'D/E Ratio', 'Industry_Avg_DE', 'Industry_Avg_ROE', 'profit_margin', 'P/B ratio', 'beta', 'P/E ratio', 'dividend_score', 'value', 'shrs_or_prn_amt']
  df[columns] = scaler.fit_transform(df[columns])
  return df

def handleOutlier(df, cols):
  for col in cols:
    Q1 = df[col].quantile(0.05)
    Q3 = df[col].quantile(0.95)
    IQR = Q3 - Q1
    multiplier = 3
    df = df[(df[col] >= Q1 - multiplier * IQR) & (df[col] <= Q3 + multiplier * IQR)]
  return df

def encodeCatData(df):
  encoder = LabelEncoder()
  df['industry'] = encoder.fit_transform(df['industry'])
  industry_mapping = {str(k): int(v) for k, v in zip(encoder.classes_, encoder.transform(encoder.classes_))}
  with open('industry_mapping.json', 'w') as f:
    json.dump(industry_mapping, f, indent=4)
  return df

def fetchRec(ticker):
  try:
    stock = yf.Ticker(ticker)
    rec = stock.recommendations

    if rec is None or rec.empty:
      print(f"No recommendations for {ticker}, assigning default score of 0.5")
      return 0.5

    score = getRecScore(rec)
    print(f"score: {score}")
    return score
  except Exception as e:
    print(f"Error for {ticker}: {e}")
    return 0.5

# 0.0 - 0.2: Strong Sell; 0.2 - 0.4: Sell; 0.4 - 0.6: Hold; 0.6 - 0.8: Buy; 0.8 - 1.0: Strong Buy
def getRecScore(rec):
  rec_map = {'strongBuy': 1, 'buy': 2, 'hold': 3, 'sell': 4, 'strongSell': 5}

  rec['mean_rec'] = (
        rec['strongBuy'] * rec_map['strongBuy'] +
        rec['buy'] * rec_map['buy'] +
        rec['hold'] * rec_map['hold'] +
        rec['sell'] * rec_map['sell'] +
        rec['strongSell'] * rec_map['strongSell']
    ) / rec[['strongBuy', 'buy', 'hold', 'sell', 'strongSell']].sum(axis=1)

  periods = rec['period'].unique()
  sorted_periods = sorted(periods, key=lambda x: int(x.replace('m', '')), reverse=True)
  weights = {period: np.exp(-0.5 * i) for i, period in enumerate(sorted_periods)}
  rec['weight'] = rec['period'].map(weights)
  weighted_score = (5 - ((rec['mean_rec'] * rec['weight']).sum() / rec['weight'].sum())) / 5

  return weighted_score

# to be improved
def getCombinedScore(df):
  weight = 0.5
  df['recommendation_score'] = df['ticker'].apply(fetchRec)
  df['combined_score'] = weight * df['recommendation_score'] + (1 - weight) * df['value']
  df['ranking_target'] = df.groupby('periodOfReport')['combined_score'].rank(ascending=False, method='dense').astype(int) - 1
  return df

def analyse():
  df = pd.read_csv(csvName)
  df = df.drop_duplicates()

  # df = handleOutdatedComp(df)
  # df = handleFundColumns(df)
  # df = handleMoreColumns(df)
  # df = handleMissingVal(df)

  # more features first run

  # df = cleanData(df)
  # df = handleOutlier(df, ['dividend_score', 'value_change_pct', 'shares_change_pct'])
  # df = scaleData(df)
  # df = encodeCatData(df)
  # df['ranking_target'] = df.groupby('periodOfReport')['value'].rank(ascending=False, method='dense').astype(int) - 1

  df.to_csv(csvName, index=False)
