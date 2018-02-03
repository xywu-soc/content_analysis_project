#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 20:13:26 2018

Content Analysis: Final Project -- Web Scraper

@author: xywu
"""

import requests
import urllib.parse
from bs4 import BeautifulSoup

def get_text(url, tag):
    '''
    '''
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')
    lst = soup.findAll(tag, soup)
    
    content = []
    for i in range(len(lst)):
        if lst[i].text != '':
            content.append(lst[i].text)
    
    return content

test_url = "http://tvmegasite.net/transcripts/aw/older/2006/aw-trans-01-03-06.shtml"
test = get_text(test_url, 'p')

def get_url(parent_url, tag1, tag2, exclude_lst):
    '''
    '''
    rv = []
    
    r = requests.get(parent_url)
    soup = BeautifulSoup(r.text, 'lxml')
    tag1_lst = soup.findAll(tag1)
    
    for item in tag1_lst:
        if item.text != 'Parent Directory':
            temp = item.get(tag2)
            if temp not in exclude_lst:
                new_url = urllib.parse.urljoin(parent_url, temp)
                rv.append(new_url)
    
    return rv

exclude_lst = ['?N=D', '?M=A', '?S=A', '?D=A']
parent_url = 'http://tvmegasite.net/transcripts/'
layer1_url = get_url(parent_url, 'a', 'href', ['?N=D', '?M=A', '?S=A', '?D=A', '_borders/', '_fpclass/', '_private/', '_themes/', 'images/', 'resources/', 'test/'])
layer2_url = [x+'older/' for x in layer1_url]
layer3_url_amc = get_url(layer2_url[0], 'a', 'href', ['?N=D', '?M=A', '?S=A', '?D=A'])
layer4_url_amc_2013 = get_url(layer3_url_amc[-1], 'a', 'href', exclude_lst)
            