#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 15:45:46 2018

Computational Content Analysis: Final Project -- Topic Modeling

This file construct a LDA topic model, and visualize important topics and character profiles.
"""

import pandas as pd
import numpy as np
import nltk
import lucem_illud
import gensim
import matplotlib.pyplot as plt
from math import pi


##### Data Preparation #####

# Read in data and clean
scene_speaker_df = pd.read_csv('aggregate_speaker_scene.csv')
scene_speaker_df['V1'] = scene_speaker_df['V1'].fillna('')

# Tokenize and normalize data
scene_speaker_df['tokenized_text'] = scene_speaker_df['V1'].apply(lambda x: nltk.word_tokenize(x))
scene_speaker_df['normalized_text'] = scene_speaker_df['tokenized_text'].apply(lambda x: lucem_illud.normalizeTokens(x, stopwordLst = lucem_illud.stop_words_basic, stemmer = lucem_illud.stemmer_basic))


##### LDA Topic Modeling #####

# Generate corpus objects
print('Start to get dictionary')
so_d = gensim.corpora.Dictionary(scene_speaker_df['normalized_text'])
so_corpus = [so_d.doc2bow(text) for text in scene_speaker_df['normalized_text']]
gensim.corpora.MmCorpus.serialize('scene_character.mm', so_corpus)
so_senmm = gensim.corpora.MmCorpus('scene_character.mm')

# Run LDA topic model
print('Topic model')
so_senlda = gensim.models.ldamodel.LdaModel(corpus=so_senmm, id2word=so_d, num_topics=30, alpha='auto', eta='auto')
so_senlda.save('new_scene_character.model') # save the model
# Load the model and check the top words for each topic
so_senlda =  gensim.models.LdaModel.load('new_scene_character.model')
import re
word_d = {'topic':[], 'words':[]}
for i in range(so_senlda.num_topics):
    top_words = so_senlda.print_topic(i, topn=30)
    word_lst = re.findall('[a-z]+', top_words)
    word_str = ', '.join(word_lst)
    word_d['topic'].append('topic_{}'.format(i))
    word_d['words'].append(word_str)
word_df = pd.DataFrame(word_d)
word_df.to_csv('topic_topwords.csv')


##### Extract Information from the Model #####

# Dictionary to hold the topic loadings for each observation
scene_speaker_df['id'] = np.arange(scene_speaker_df.shape[0])
scene_speaker_so_ldaDF = pd.DataFrame({
        'name' : scene_speaker_df['id'],
        'topics' : [so_senlda[so_d.doc2bow(l)] for l in scene_speaker_df['normalized_text']]
    })

# Dictionary to temporally hold the probabilities
scene_speaker_topicsProbDict = {i : [0] * len(scene_speaker_so_ldaDF) for i in range(so_senlda.num_topics)}
# Load them into the dict
for index, topicTuples in enumerate(scene_speaker_so_ldaDF['topics']):
    for topicNum, prob in topicTuples:
        scene_speaker_topicsProbDict[topicNum][index] = prob
# Update the DataFrame
for topicNum in range(so_senlda.num_topics):
    scene_speaker_so_ldaDF['topic_{}'.format(topicNum)] = scene_speaker_topicsProbDict[topicNum]

# Get the top words for each topic
scene_speaker_so_topicsDict = {}
for topicNum in range(so_senlda.num_topics):
    topicWords = [w for w, p in so_senlda.show_topic(topicNum, topn=30)]
    scene_speaker_so_topicsDict['Topic_{}'.format(topicNum)] = topicWords

# Turn the information into a Pandas DataFrame
scene_speaker_so_wordRanksDF = pd.DataFrame(scene_speaker_so_topicsDict)
scene_speaker_so_wordRanksDF

# Add gender information to the DataFrame
df_new = pd.concat([scene_speaker_df, scene_speaker_so_ldaDF], axis=1)
df_gender_show = pd.read_csv('aggregate_speaker_scene_new.csv')
df_new = pd.merge(df_new, df_gender_show, how='inner', on=['date_scene', 'opera_name', 'speaker', 'V1'])


##### Topic labeling (Manually Check Top Words and Texts of all the Topics)#####

# Define a function to show the texts of each topic for interpretation
def show_topic_content(df, topic_num):
    t = df.nlargest(30, 'topic_{}'.format(topic_num))['V1']
    for i in range(t.shape[0]):
        print(t.iloc[i])
        print()

# How to use check the top words and texts
#so_wordRanksDF['Topic_0']
#show_topic_content(df_new, 0)


##### Comparison between Gender #####

# Define a function to calcualte the mean for each topic over the whole corpus
def get_mean(df, topic_num):
    loc = topic_num + 12
    topic = df.iloc[:, loc].values
    topic_avg = topic.mean()
    return topic_avg

# Since some of the characters does not have gender labeled, get a subset of
#    data for the comparison
gender = ['M', 'F']
df_subset = df_new[df_new['gender'].isin(gender)]

# Calculate the mean for each topic in the whole corpus
topic_mean_lst = []
for i in range(30):
    temp = get_mean(df_subset, i)
    topic_mean_lst.append(temp)

# Get mean loading for each topic by gender
df_new_gender = df_new.groupby(['gender'])['topic_0', 'topic_1', 'topic_2', \
                            'topic_3', 'topic_4', 'topic_5', 'topic_6', \
                            'topic_7', 'topic_8', 'topic_9', 'topic_10', \
                            'topic_11', 'topic_12', 'topic_13', 'topic_14', \
                            'topic_15', 'topic_16', 'topic_17', 'topic_18', \
                            'topic_19', 'topic_20', 'topic_21', 'topic_22', \
                            'topic_23', 'topic_24', 'topic_25', 'topic_26', \
                            'topic_27', 'topic_28', 'topic_29'].mean()

# Get proportion: gender mean on average
topic_female = np.array(df_new_gender.iloc[0])/np.array(topic_mean_lst)
topic_male = np.array(df_new_gender.iloc[1])/np.array(topic_mean_lst)
gender_diff_new = []
for i in range(30):
    if topic_female[i] > 1:
        temp = topic_female[i] - 1
        gender_diff_new.append(temp)
    elif topic_male[i] > 1:
        temp = (topic_male[i] - 1) * (-1)
        gender_diff_new.append(temp)

# Compute the proportions with 1 to get ready for topic labeling
diff_male = []
diff_male_index = []
diff_female = []
diff_female_index = []
rm_lst = [1, 2, 7, 8, 12, 15, 16, 18, 21, 25, 28]
for i in range(len(gender_diff_new)):
    if i in rm_lst:
        pass
    else:
        if gender_diff_new[i] < 0:
            diff_male_index.append(i)
            diff_male.append(gender_diff_new[i])
        elif gender_diff_new[i] >= 0:
            diff_female_index.append(i)
            diff_female.append(gender_diff_new[i])
        
### Plot ###     
fig, ax = plt.subplots(1,1,figsize=(15,15))#
plt.hlines(0, 0, 29, colors='gray', linestyles='--')
plt.bar(np.array(diff_male_index), np.array(diff_male), color='mediumseagreen', label='Male')
plt.bar(diff_female_index, diff_female, color='darkorange', label='Female')
# Invert the plot to make the male on the left
ax.invert_yaxis()
# Set ticks and labels
ax.set_xticks(np.arange(30))
ax.xaxis.tick_top()
ax.set_xticks(np.arange(30))
ax.set_yticks([-0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14])
ax.set_xticklabels(np.arange(30), rotation=270, fontsize=16)
ax.set_yticklabels([-0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14], rotation=270, fontsize=16)

    
##### Character Profiles: Radius Plot for Characters #####

# Get topic
topic_mean = []
for i in range(30):
    temp = get_mean(df_new, i)
    topic_mean.append(temp)

# Get the list of top characters
df_characters50 = list(pd.DataFrame(df_new['speaker'].value_counts()[:50]).index)

# Get the mean loading for each topic of by character
df_speaker = df_new.groupby(['speaker'])['topic_0', 'topic_1', 'topic_2', \
                            'topic_3', 'topic_4', 'topic_5', 'topic_6', \
                            'topic_7', 'topic_8', 'topic_9', 'topic_10', \
                            'topic_11', 'topic_12', 'topic_13', 'topic_14', \
                            'topic_15', 'topic_16', 'topic_17', 'topic_18', \
                            'topic_19', 'topic_20', 'topic_21', 'topic_22', \
                            'topic_23', 'topic_24', 'topic_25', 'topic_26', \
                            'topic_27', 'topic_28', 'topic_29'].mean()


# Define a function to get mean topic loadings for an assigned character
def get_character_mean(df, character_name, topic_lst, rm_topic_lst, topic_mean):
    '''
    This function computes the mean loading of each topic within an assigned
    character. And remove the topics from the results according to the
    imported list.
    
    Inputs:
        df: a Pandas DataFrame, containing statement information and topic
            loadings
        character_name: a string, name for the character
        topic_lst: a list of topics as strings
        rm_topic_lst: a list of indices, indicating the topics to remove
        topic_mean: a list of average topic loadings as floating numbers, of
            the whole corpus
    
    Output:
        labels: a list of strings, labels of the topics
        character_avg: a list of average topic loadings as floating numbers
    '''      
    # Get average topic loadings
    character_df = df[df['speaker']==character_name]
    character_avg = character_df[['topic_0', 'topic_1', 'topic_2', \
                            'topic_3', 'topic_4', 'topic_5', 'topic_6', \
                            'topic_7', 'topic_8', 'topic_9', 'topic_10', \
                            'topic_11', 'topic_12', 'topic_13', 'topic_14', \
                            'topic_15', 'topic_16', 'topic_17', 'topic_18', \
                            'topic_19', 'topic_20', 'topic_21', 'topic_22', \
                            'topic_23', 'topic_24', 'topic_25', 'topic_26', \
                            'topic_27', 'topic_28', 'topic_29']].mean()
    
    # Get the proportions
    character_diff = list(np.array(character_avg)/topic_mean)
    
    # Remove uninterpretable topics
    labels = []
    vals = []
    for i in range(30):
        if i in rm_topic_lst:
            pass
        else:
            labels.append(topic_lst[i])
            vals.append(character_diff[i])
    
    return labels, vals


# Get a function to plot the radar plot
def get_radar_plot(character_name, labels, character_vals, colors, is_lst=0):
    '''
    This functions draws radar plot for character profile(s).
    
    Inputs:
        character_name: a string or a list of strings, name(s) of the character
        labels: a list of strings, labels for topics
        character_vals: a list of floating numbers or a list of list of floating numbers,
            containing topic loadings of the character(s)
        colors: a string or a list of strings to indicate numbers of plot
        is_lst: 0 = the vals is a list of numbers, 1= the vals is a list of
            lists. Default is 0.
    
    Outputs:
        None, just plot
    '''
    # Data preparation
    if is_lst == 0:
        # Get N to calculate angles
        N = len(character_vals)
        # Make the plot closed
        vals = character_vals + character_vals[:1]
    else:
        # Get N to calculate angles
        vals = []
        N = len(character_vals[0])
        for i in range(len(vals)):
            temp = character_vals[i] + character_vals[i][:1]
            vals.append(temp)
    
    # Get the angles
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
        

    # Initialize the spider plot
    ax = plt.subplot(111, polar=True)
    plt.title('Character Profile: {}'.format(str(character_name)), y=1.08)
    # Draw one axe per variable and add labels
    plt.xticks(angles[:-1], labels, color='grey', size=8)
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([1.0], '1.0')
    plt.ylim([0,3])
    # Plot data
    if is_lst == 0:
        ax.plot(angles, vals, color=colors[0], linewidth=1, linestyle='solid')
        ax.fill(angles, vals, color=colors[1], alpha=0.5)
    else:
        for i in range(len(vals)):
            ax.plot(angles, vals[i], color=colors[i][0], linewidth=1, linestyle='solid')
            ax.fill(angles, vals[i], color=colors[i][1], alpha=0.5)
    
    return None


# Label of the topics
categories = ['Talk about Family', 'topic_1', 'topic_2', \
                            'Relationships', 'Love', 'Judiciary', 'Begging', \
                            'topic_7', 'topic_8', 'Celebration', 'Rage', \
                            'Profession', 'topic_12', 'Talk within Family', \
                            'People', 'topic_15', 'topic_16', 'Emotions', \
                            'topic_18', 'Laughter', 'Leisure', 'topic_21', \
                            'Crime', 'Urge', 'Christmas', 'topic_25', 'Crime', \
                            'Birth & Death', 'topic_28', 'Calming']


# Get names: check the list of 50 most frequent characters
#df_characters50

# Check  Alexis(GH main character) !!!!! crime person
al_labels, al_vals = get_character_mean(df_new, 'Alexis', categories, rm_lst, topic_mean)
get_radar_plot('Alexis', al_labels, al_vals, ['darkorange', 'orange'])

# Check Georgie (GH main character) !!!!! family lady
g_labels, g_vals = get_character_mean(df_new, 'Georgie', categories, rm_lst, topic_mean)
get_radar_plot('Georgie', g_labels, g_vals, ['darkorange', 'orange'])

# Check Nikolas !!!!! crime guy
nic_labels, nic_vals = get_character_mean(df_new, 'Nikolas', categories, rm_lst, topic_mean)
get_radar_plot('Nikolas', nic_labels, nic_vals, ['darkgreen', 'green'])

# Check Alexander !!!!! rage guy
alex_labels, alex_vals = get_character_mean(df_new, 'Alexander', categories, rm_lst, topic_mean)
get_radar_plot(alex_labels, alex_vals, ['darkgreen', 'green'])

