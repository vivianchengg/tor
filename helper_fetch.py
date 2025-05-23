#! /usr/bin/env python3
import time

import requests, re
import random
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import xml.etree.ElementTree as ET
import yfinance as yf

headers = {
    'User-Agent': 'casual project: side project trying to work with web scraping'
}

secUrl = 'https://www.sec.gov'
archiveUrl = '/Archives/edgar/data/1067983/'
targetOwner = 'Berkshire Hathaway Inc'

def fetchLink(url):
  res = requests.get(url, headers=headers)
  soup = BeautifulSoup(res.text, 'html.parser')
  return soup

def fetchXmlContent(url, reportDate, allData):
  print(f"\nfetching content from {url}...")
  res = requests.get(url, headers=headers)
  tree = ET.ElementTree(ET.fromstring(res.content))
  root = tree.getroot()
  df = parseXml(root, reportDate)

  print(df)
  allData = pd.concat([allData, df], ignore_index=True)
  return allData

def parseXml(element, period_of_report):
  data = {}
  namespace = {'ns': 'http://www.sec.gov/edgar/document/thirteenf/informationtable'}

  for info in element.findall('.//ns:infoTable', namespace):
    name_of_issuer = info.find('ns:nameOfIssuer', namespace).text if info.find('ns:nameOfIssuer', namespace) is not None else None
    title_of_class = info.find('ns:titleOfClass', namespace).text if info.find('ns:titleOfClass', namespace) is not None else None
    cusip = info.find('ns:cusip', namespace).text if info.find('ns:cusip', namespace) is not None else None
    value = int(info.find('ns:value', namespace).text) if info.find('ns:value', namespace) is not None else None
    shrs_or_prn_amt = int(info.find('ns:shrsOrPrnAmt/ns:sshPrnamt', namespace).text) if info.find('ns:shrsOrPrnAmt/ns:sshPrnamt', namespace) is not None else None

    if name_of_issuer and title_of_class and cusip and value and shrs_or_prn_amt:
        key = (cusip, name_of_issuer, title_of_class, period_of_report)
        if key in data:
          data[key]['value'] += value
          data[key]['shrs_or_prn_amt'] += shrs_or_prn_amt
        else:
          data[key] = {
            'cusip': cusip,
            'nameOfIssuer': name_of_issuer,
            'titleOfClass': title_of_class,
            'periodOfReport': period_of_report,
            'value': value,
            'shrs_or_prn_amt': shrs_or_prn_amt
          }

  return pd.DataFrame.from_dict(data, orient='index')

def checkPrimary(url):
  res = requests.get(url, headers=headers)
  xmlContent = res.text
  tree = ET.ElementTree(ET.fromstring(xmlContent))
  root = tree.getroot()
  namespace = {'ns': 'http://www.sec.gov/edgar/thirteenffiler'}
  ownerName = root.find('.//ns:filingManager/ns:name', namespace)
  reportDate = root.find('.//ns:periodOfReport', namespace)

  return (ownerName is not None and ownerName.text == targetOwner), (reportDate.text if reportDate is not None else None)

def checkTarget(soup):
  isTargetXml = False
  isTargetOwner = False
  xmlUrl = ''
  reportDate = ''

  for link in soup.find_all('a'):
    href = link.get('href')

    # check target xml
    if href and href.endswith('.xml'):
      xmlName = href.split('/')[-1]
      # digit or '-' only
      regex = r'^[\d-]+.xml$'
      if re.search(regex, xmlName):
        xmlUrl = f"{secUrl}{href}"
        isTargetXml = True

      # check target owner
      if xmlName == 'primary_doc.xml':
        pDocUrl = f"{secUrl}{href}"
        isTargetOwner, reportDate = checkPrimary(pDocUrl)

    if isTargetXml and isTargetOwner:
      print('FOUND')
      break

  return xmlUrl, (isTargetXml and isTargetOwner), reportDate

def fetchFiles(allData):
  soup = fetchLink(f"{secUrl}{archiveUrl}")

  for link in soup.find_all('a'):
    href = link.get('href')
    if not href or not href.startswith(archiveUrl):
      continue

    folderUrl = f"{secUrl}{href}"
    print(f"checking folder {folderUrl}...")
    soup = fetchLink(folderUrl)
    xmlUrl, isTarget, reportDate = checkTarget(soup)

    if isTarget:
      allData = fetchXmlContent(xmlUrl, reportDate, allData)

    time.sleep(0.11) # SEC’s limit of 10 requests per second

  print('done')
  return allData