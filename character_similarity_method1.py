#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 22:07:12 2018

Computational Content Analysis: Final Project -- Character Similarity Method 1

"""

import pandas as pd
import numpy as np
import sklearn
import lucem_illud
from nltk import word_tokenize
import matplotlib.pyplot as plt


def get_top(df, end_index):
    network = list(pd.DataFrame(df['speaker'].value_counts()[:end_index]).index)
    gender = []
    for j in range(len(network)):
        for i in range(df.shape[0]):
            if df.iloc[i]['speaker'] == network[j]:
                gender.append(df.iloc[i]['gender'])
                break
    return network, gender


def get_text(df, names, gender):
    texts = []
    df['speech'] = df['speech'].fillna('')
    for i in range(len(names)):
        print(i)
        temp_df = df[df['speaker']==names[i]].groupby(['speaker'])['speech'].apply(lambda x: "%s" % ', '.join(x))
        statement = temp_df[names[i]]
        tokenized = word_tokenize(statement)
        normalized = lucem_illud.normalizeTokens(tokenized, stopwordLst = lucem_illud.stop_words_basic, stemmer = lucem_illud.stemmer_basic)
        texts.append(normalized)
    
    new_df = pd.DataFrame({'speaker':names, 'gender':gender, 'normalized_text':texts})
    return new_df
            

def calculate_cos(df_all):   
    # Create empty dictionaries
    unigrams_all = {}
    unigrams_statement = {}
    
    for i in range(df_all.shape[0]):
        print(i)
        unigrams_statement[df_all.iloc[i]['speaker']] = [df_all.iloc[i]['gender'], {}]
        for word in df_all.iloc[i]['normalized_text']:
            unigrams_all[word] = unigrams_all.get(word, 0) + 1
            unigrams_statement[df_all.iloc[i]['speaker']][1][word] = unigrams_statement[df_all.iloc[i]['speaker']][1].get(word, 0) + 1
        
    unigrams_sorted = []
    for item in sorted(unigrams_all, key=unigrams_all.get, reverse=True):
        unigrams_sorted.append(item)
    
    m = []
    m_tf = []
    for k,v in unigrams_statement.items():
        temp = []
        temp.append(v[0])
        temp.append(k)
        temp_tf = []

        for j in range(len(unigrams_sorted)):
            if unigrams_sorted[j] in v[1].keys():
                temp.append(v[1][unigrams_sorted[j]])
                temp_tf.append(v[1][unigrams_sorted[j]])
#                temp_tf.append((v[1][unigrams_sorted[j]]) / sum(v[1].values()))
            else:
                temp.append(0)
                temp_tf.append(0)
        m.append(temp)
        m_tf.append(temp_tf)
    
    # Calculate idf
    doc_has_term = []
    for i in range(len(unigrams_sorted)):
        temp = 0
        for j in range(len(m)):
            if m[j][i+2] != 0:
                temp += 1
        doc_has_term.append(temp)
    idf = np.log(np.full_like(doc_has_term, 30) / np.array(doc_has_term))
    # Calculate tf-idf
    tf_idf = np.array(m_tf) * idf
    
    # Get cosine similarity with tf-idf
    cos_all = sklearn.metrics.pairwise.cosine_similarity(tf_idf)
    
    # Get average similarity within gender
    male_index = []
    female_index = []
    for i in range(len(m)):
        if m[i][0] == 'F':
            female_index.append(i)
        else:
            male_index.append(i)

    within_male_cos = []
    within_female_cos = []
    across_cos = []
    for i in range(len(male_index)):
        for j in range(len(male_index)):
            if i != j:
                within_male_cos.append(cos_all[i][j])
    for i in range(len(female_index)):
        for j in range(len(female_index)):
            if i != j:
                within_female_cos.append(cos_all[i][j])
    for i in range(len(male_index)):
        for j in range(len(female_index)):
            across_cos.append(cos_all[i][j])

    # within gender similarity
    within_male = np.mean(within_male_cos)
    within_female = np.mean(within_female_cos)
    # between gender similarity
    across_gender = np.mean(across_cos)

    rv = [within_male, within_female, across_gender]
    
    return rv, m, cos_all


def clean_network(network, gender):
    network_final = []
    gender_final = []
    for i in range(len(gender)):
        if gender[i] == 'M' or gender[i] == 'F':
            network_final.append(network[i])
            gender_final.append(gender[i])
    rv = [network_final, gender_final]
    return rv


##### Plot for AMC #####
    
### Top 50 ###
amc_df = pd.read_csv('amc_ml.csv')
amc_top, amc_top_gender = get_top(amc_df, 51)
amc_network = clean_network(amc_top, amc_top_gender)
amc_text = get_text(amc_df, amc_network[0], amc_network[1])
amc_avg, amc_m, amc_cos = calculate_cos(amc_text)

amc_names_all = []
for i in range(len(amc_m)):
    temp = '{}({})'.format(amc_m[i][1], amc_m[i][0])
    amc_names_all.append(temp)

fig, ax = plt.subplots(figsize=(10, 7))
hmap = ax.pcolor(amc_cos) #, cmap='terrain'
cbar = plt.colorbar(hmap)
ax.set_xticks(np.arange(amc_cos.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(amc_cos.shape[0]) + 0.5, minor=False)
ax.set_xticklabels(amc_names_all, minor=False, rotation=270, fontsize=9)
ax.set_yticklabels(amc_names_all, minor=False, fontsize=9)
ax.set_title('Cosine Similarity Between Top-50 Characters: AMC', fontsize=12)

### Top 20 ###
amc_top2, amc_top_gender2 = get_top(amc_df, 20)
#amc_network = clean_network(amc_top, amc_top_gender)
amc_text2 = get_text(amc_df, amc_top2, amc_top_gender2)
amc_avg2, amc_m2, amc_cos2 = calculate_cos(amc_text2)

amc_names_all2 = []
for i in range(len(amc_m2)):
    temp = '{}({})'.format(amc_m2[i][1], amc_m2[i][0])
    amc_names_all2.append(temp)
amc_names_all2[5] = 'J.R.(M)'

fig, ax = plt.subplots(figsize=(10, 7))
hmap = ax.pcolor(amc_cos2) #, cmap='terrain'
cbar = plt.colorbar(hmap)
ax.set_xticks(np.arange(amc_cos2.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(amc_cos2.shape[0]) + 0.5, minor=False)
ax.set_xticklabels(amc_names_all2, minor=False, rotation=270, fontsize=12)
ax.set_yticklabels(amc_names_all2, minor=False, fontsize=12)
ax.set_title('Cosine Similarity Between Top-20 Characters: AMC', fontsize=16)


##### Plot for GH #####
gh_df = pd.read_csv('gh_ml.csv')
gh_top, gh_top_gender = get_top(gh_df, 50)
gh_network = clean_network(gh_top, gh_top_gender)
gh_text = get_text(gh_df, gh_network[0], gh_network[1])
gh_avg, gh_m, gh_cos = calculate_cos(gh_text)

gh_names_all = []
for i in range(len(gh_m)):
    temp = '{}({})'.format(gh_m[i][1], gh_m[i][0])
    gh_names_all.append(temp)

fig, ax = plt.subplots(figsize=(10, 7))
hmap = ax.pcolor(gh_cos) #, cmap='terrain'
cbar = plt.colorbar(hmap)
ax.set_xticks(np.arange(gh_cos.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(gh_cos.shape[0]) + 0.5, minor=False)
ax.set_xticklabels(gh_names_all, minor=False, rotation=270, fontsize=9)
ax.set_yticklabels(gh_names_all, minor=False, fontsize=9)
ax.set_title('Cosine Similarity Between Top-50 Characters: GH')

### Top 20 ###
gh_top2, gh_top_gender2 = get_top(gh_df, 20)
#gh_network = clean_network(gh_top, gh_top_gender)
gh_text2 = get_text(gh_df, gh_top2, gh_top_gender2)
gh_avg2, gh_m2, gh_cos2 = calculate_cos(gh_text2)

gh_names_all2 = []
for i in range(len(gh_m2)):
    temp = '{}({})'.format(gh_m2[i][1], gh_m2[i][0])
    gh_names_all2.append(temp)

fig, ax = plt.subplots(figsize=(10, 7))
hmap = ax.pcolor(gh_cos2) #, cmap='terrain'
cbar = plt.colorbar(hmap)
ax.set_xticks(np.arange(gh_cos2.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(gh_cos2.shape[0]) + 0.5, minor=False)
ax.set_xticklabels(gh_names_all2, minor=False, rotation=270, fontsize=12)
ax.set_yticklabels(gh_names_all2, minor=False, fontsize=12)
ax.set_title('Cosine Similarity Between Top-20 Characters: GH', fontsize=16)


##### Plot for Days of Our Lives #####
days_df = pd.read_csv('days_ml.csv')
days_top, days_top_gender = get_top(days_df, 50)
days_network = clean_network(days_top, days_top_gender)
days_text = get_text(days_df, days_network[0], days_network[1])
days_avg, days_m, days_cos = calculate_cos(days_text)

days_names_all = []
for i in range(len(days_m)):
    temp = '{}({})'.format(days_m[i][1], days_m[i][0])
    days_names_all.append(temp)

fig, ax = plt.subplots(figsize=(10, 7))
hmap = ax.pcolor(days_cos) #, cmap='terrain'
cbar = plt.colorbar(hmap)
ax.set_xticks(np.arange(days_cos.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(days_cos.shape[0]) + 0.5, minor=False)
ax.set_xticklabels(days_names_all, minor=False, rotation=270, fontsize=9)
ax.set_yticklabels(days_names_all, minor=False, fontsize=9)
ax.set_title('Cosine Similarity Between Top-50 Characters: DoOL')

### Top 20 ###
days_top2, days_top_gender2 = get_top(days_df, 20)
#days_network2 = clean_network(days_top, days_top_gender)
days_text2 = get_text(days_df, days_top2, days_top_gender2)
days_avg2, days_m2, days_cos2 = calculate_cos(days_text2)

days_names_all2 = []
for i in range(len(days_m2)):
    temp = '{}({})'.format(days_m2[i][1], days_m2[i][0])
    days_names_all2.append(temp)

fig, ax = plt.subplots(figsize=(10, 7))
hmap = ax.pcolor(days_cos2) #, cmap='terrain'
cbar = plt.colorbar(hmap)
ax.set_xticks(np.arange(days_cos2.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(days_cos2.shape[0]) + 0.5, minor=False)
ax.set_xticklabels(days_names_all2, minor=False, rotation=270, fontsize=12)
ax.set_yticklabels(days_names_all2, minor=False, fontsize=12)
ax.set_title('Cosine Similarity Between Top-20 Characters: DoOL', fontsize=16)


##### Save matrices #####
np.savetxt('amc_top20.csv', amc_cos2, delimiter=',')
np.savetxt('gh_top20.csv', gh_cos2, delimiter=',')
np.savetxt('days_top20.csv', days_cos2, delimiter=',')


##### Calculate Within-Group and Between-Group Similarity

### Top 50 ###

# AMC: amc_cos, amc_top_gender
amc_male = []
amc_female = []
for i in range(50):
    if amc_top_gender[i] == 'F':
        amc_female.append(amc_cos[i])
    else:
        amc_male.append(amc_cos[i])
amc_male_mean = np.mean(np.array(amc_male))
amc_female_mean = np.mean(np.array(amc_female))

# GH: gh_cos, gh_top_gender
gh_male = []
gh_female = []
for i in range(50):
    if gh_top_gender[i] == 'F':
        gh_female.append(gh_cos[i])
    else:
        gh_male.append(gh_cos[i])
gh_male_mean = np.mean(np.array(gh_male))
gh_female_mean = np.mean(np.array(gh_female))

# Days: days_cos, days_top_gender
days_male = []
days_female = []
for i in range(50):
    if days_top_gender[i] == 'F':
        days_female.append(days_cos[i])
    else:
        days_male.append(days_cos[i])
days_male_mean = np.mean(np.array(days_male))
days_female_mean = np.mean(np.array(days_female))


### Top 20 ###

# AMC: amc_cos2, amc_top_gender2
amc_male2 = []
amc_female2 = []
for i in range(20):
    if amc_top_gender2[i] == 'F':
        amc_female2.append(amc_cos2[i])
    else:
        amc_male2.append(amc_cos2[i])
amc_male_mean2 = np.mean(np.array(amc_male2))
amc_female_mean2 = np.mean(np.array(amc_female2))

# GH: gh_cos2, gh_top_gender2
gh_male2 = []
gh_female2 = []
for i in range(20):
    if gh_top_gender2[i] == 'F':
        gh_female2.append(gh_cos2[i])
    else:
        gh_male2.append(gh_cos2[i])
gh_male_mean2 = np.mean(np.array(gh_male2))
gh_female_mean2 = np.mean(np.array(gh_female2))

# days: days_cos2, days_top_gender2
days_male2 = []
days_female2 = []
for i in range(20):
    if days_top_gender2[i] == 'F':
        days_female2.append(days_cos2[i])
    else:
        days_male2.append(days_cos2[i])
days_male_mean2 = np.mean(np.array(days_male2))
days_female_mean2 = np.mean(np.array(days_female2))