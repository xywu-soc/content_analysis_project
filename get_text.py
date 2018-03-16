#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 22:12:41 2018

Computational Content Analysis: Final Project -- Texts Scraper

This file is to scrape the transcripts of soap operas from webpages. It uses the
url.csv file constructed in 'get_url.py'. And it stores the scraped texts in
.txt files by year and by show.
"""

import pandas as pd
import requests
import time
import bs4
import re
import os


def get_text(url):
    rv = []
    
    time.sleep(1)
    r = requests.get(url)
    soup = bs4.BeautifulSoup(r.text, 'lxml')
    
    if soup.body != '':
        pTags = soup.body.findAll('p')
    else:
        pTags = soup.findAll('p')
    
    for i in range(len(pTags)):
        temp1 = pTags[i].text.replace('\r\n', ' ')
        temp2 = temp1.replace('\n', ' ')
        rv.append(temp2)
    
    return rv


def store_text(lst, folder1, folder2, file_name):
    directory = 'data/{}/{}'.format(folder1, folder2)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open('{}/{}.txt'.format(directory, file_name), 'w') as f:
        f.write('\n'.join(lst))


def main(start):
    df = pd.read_csv('urls.csv')
    
    if start == None:
        iterate_range = (0, df.shape[0])
    else:
        iterate_range = (start, df.shape[0])
        
    for i in range(iterate_range[0], iterate_range[1]):
        url = df.url.iloc[i]
        soap_opera = df.tv_name.iloc[i]
        play_year = df.year.iloc[i]
        episode = re.search('([0-9]{2}\-[0-9]{2}\-[0-9]{2})', url)
        if episode != None:
            file_name = episode.group(0)

            texts = get_text(url)
            store_text(texts, soap_opera, play_year, file_name)
        
    return texts


#main()
#main(3444)
main(8154)
