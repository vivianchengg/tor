#! /usr/bin/env python3

import os
import pandas as pd

from helper_analyse import analyse
from helper_fetch import fetchFiles
from helper_save import saveData

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
