#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 14:25:43 2018

Parsing the plain texts.

This file takes the raw .txt files of data scraped from the internet, and get
them into the format ready for analysis.
"""


import pandas as pd
import re
import os


def parse_text(opera_name):
    '''
    '''
    d = {'filename':[], 'text':[], 'speaker':[], 'speech':[], 'show_year':[], 'opera_name':[], 'show_date':[], 'scene':[], 'turn':[]}
    dir_gh = '/Users/hsswyx/Desktop/content_analysis/content_analysis_project/data/{}'.format(opera_name)
    year_lst = os.listdir(dir_gh)[1:]
    for i in range(len(year_lst)):
        if int(year_lst[i]) > 2017 or int(year_lst[i]) < 2000:
            pass
        else:
            episodes = os.listdir(dir_gh+'/'+year_lst[i])
            for j in range(len(episodes)):
                file_name = year_lst[i]+'/'+episodes[j]
                text = ''
                with open(dir_gh+'/'+file_name) as f:
                    y = 1 # scene number
                    w = 0 # speech number inside of scenes
                    flag = 1
                    for line in f:
                        if line == '\n':
                            if flag == 1:
                                continue
                            else:
                                y += 1
                                w = 0
                                flag = 1
                        else:
                            info = re.search('([A-Z].*)\:(.*)', line.replace('\n', ''))
                            if info != None:
                                speaker = info.group(1).strip()
                                speech = info.group(2).strip()
                                text = text + ' ' + speech
                                if 'Main Navigation' not in speaker:
                                    w += 1
                                    d['filename'].append(file_name)
                                    d['opera_name'].append(opera_name)
                                    d['show_year'].append(year_lst[i])
                                    d['show_date'].append(episodes[j].strip('.txt'))
                                    d['scene'].append(y)
                                    d['turn'].append(w)
                                    d['text'].append(line.replace('\n', ''))
                                    d['speaker'].append(speaker)
                                    d['speech'].append(speech)
                                    flag = 0

    df = pd.DataFrame(d)
    df = df[['filename', 'opera_name', 'show_year', 'show_date', 'scene', 'turn', 'text', 'speaker', 'speech']]
    output_name = 'parsed_{}.csv'.format(opera_name)
    df.to_csv(output_name)

    return df

dir_all_opera = '/Users/hsswyx/Desktop/content_analysis/content_analysis_project/data'
all_opera = os.listdir(dir_all_opera)[1:]

for item in all_opera:
    df = parse_text(item)
    
df_amc = parse_text('amc')

#i = 0
#while 200000*i < df_gh.shape[0]:
#    i += 1
#    temp_file_name = 'parsed_general_hospital{}.csv'.format(i)
#    df_gh.iloc[100000*i:100000*(i+1)].to_csv(temp_file_name)
#    print(i)
#df_gh.iloc[100000*9:].to_csv('parsed_general_hospital9.csv')
