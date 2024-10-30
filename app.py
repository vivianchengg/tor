#! /usr/bin/env python3

import os
import pandas as pd

from helper_analyse import analyse
from helper_fetch import fetchFiles
from helper_save import saveData

def getTickerMap():
  df1 = pd.read_csv('ticker1.txt', delimiter='|')
  df2 = pd.read_csv('ticker2.txt', delimiter='|')
  df3 = pd.read_csv('ticker3.txt', delimiter='|')
  df = pd.concat([df1, df2], ignore_index=True)
  df = pd.concat([df, df3], ignore_index=True)
  df = df[['CUSIP', 'SYMBOL', 'DESCRIPTION']]
  df.to_csv('tickerData.csv', index=False)

def main():
  # getTickerMap()
  # allData = pd.DataFrame()
  # # manage csv
  # allData = fetchFiles(allData)
  # allData = saveData(allData)

  # analyse csv
  analyse()

  # program end alert sound x3
  # for i in range(3):
  #   os.system('afplay /System/Library/Sounds/Glass.aiff')

if __name__ == '__main__':
  main()
