#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 20:13:26 2018

Computational Content Analysis: Final Project -- Web Scraper to Get All the Urls

Transcripts of each episode are stored in seperate webpages. We use this file
to build a scraper to get all the urls of the webpages. And export the urls
to a csv file.

"""

import requests
from bs4 import BeautifulSoup
import urllib.parse
import csv


# Define the function to get urls
def get_url(parent_url, tag1, tag2, exclude_lst):
    '''
    Inputs:
        parent_url: a string, the parent url that contains many next-level urls
        tag1: a string, tag in html
        tag2: a string, tag in html
        exclude_lst: a list of urls that we should avoid
    
    Outputs:
        names: a list of strings, names of each folder
        urls: a list of urls
    '''
    names = []
    urls = []
    
    r = requests.get(parent_url)
    soup = BeautifulSoup(r.text, 'lxml')
    tag1_lst = soup.findAll(tag1)
    
    for item in tag1_lst:
        if item.text != 'Parent Directory':
            temp = item.get(tag2)
            if temp not in exclude_lst:
                names.append(temp.strip('/'))
                new_url = urllib.parse.urljoin(parent_url, temp)
                urls.append(new_url)
    
    return names, urls
         

# Initialize
d = {}   
exclude_lst = ['?N=D', '?M=A', '?S=A', '?D=A']
parent_url = 'http://tvmegasite.net/transcripts/'

# Get the 1st layer
layer1_name, layer1_url = get_url(parent_url, 'a', 'href', exclude_lst+['_borders/', '_fpclass/', '_private/', '_themes/', 'images/', 'resources/', 'test/'])

# Get the 2nd layer
layer2_url = [x+'older/' for x in layer1_url]

# Get the 3rd and 4th layer
for i in range(len(layer2_url)):
    d[layer1_name[i]] = {}
    layer3_name_temp, layer3_url_temp = get_url(layer2_url[i], 'a', 'href', exclude_lst)
    for j in range(len(layer3_name_temp)):
        d[layer1_name[i]][layer3_name_temp[j]] = {}
        layer4_name_temp, layer4_url_temp = get_url(layer3_url_temp[j], 'a', 'href', exclude_lst+['favicon.ico', 'template.shtml'])
        for k in range(len(layer4_name_temp)):
            d[layer1_name[i]][layer3_name_temp[j]] = layer4_url_temp


# Preparation to write all the urls into a csv file
column_names = ['tv_name', 'year', 'url']
all_urls = []
for k1, v1 in d.items():
    for k2, v2 in v1.items():
        for i in range(len(v2)):
            temp = [k1, k2, v2[i]]
            all_urls.append(temp)
# Insert column names
all_urls.insert(0, column_names)


# Write the csv file
with open ('urls.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(all_urls)
