#! /usr/bin/env python3

import os
import pandas as pd

from helper_analyse import analyse
from helper_fetch import fetchFiles
from helper_save import saveData

headers = {
    'User-Agent': 'casual project: side project trying to work with web scraping'
}


secUrl = 'https://www.sec.gov'
archiveUrl = '/Archives/edgar/data/1067983/'
targetOwner = 'Berkshire Hathaway Inc'
csvName = 'berkshire_holdings_combined.csv'

def main():
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