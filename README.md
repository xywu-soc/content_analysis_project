# Computational Content Analysis Project: Gender Roles in American Soap Operas

Team members: Kwon, Stuhler, and Wu

This is the GitHub repository for our final project of the Computational Content Analysis class. With the transcripts of All My Children, General Hospital, and Days of Our Lives, we analyze gender roles in American Soap Operas.

This is the README file of our annotated scripts.


## Part I: Data Scraping

### get_url.py
Web scraper to get all the urls. Transcripts of each episode are stored in seperated webpages. We use this file to build a scraper to get all the urls of the webpages. And export the urls to a csv file.

### get_text.py
Text scraper. This file is to scrape the transcripts of soap operas from webpages. It uses the url.csv file constructed in 'get_url.py'. And it stores the scraped texts in .txt files by year and by show.


## Part II: Data Processing

### parse_text.py
Parse the plain texts. This file takes the raw .txt files of data scraped from the internet, and get them into the format ready for analysis.


## Part III: Data Analysis

### 1. Interactive Network Analysis (Analysis Part 1)
#### Network_script.R
This file runs the network analysis of characters based on their speaking turns in the soap operas.

### 2. Topic Modeling: LDA (Analysis Part 2)
#### topic_scene_character.py
This file runs the LDA topic model and its visualization.

### 3. Character Complexity (Analysis Part 3)
These files contain within-character complexity. They install and import the pre-trained Word2vec models, project each statement of each character into the vector space, aggregate each word vector into centroid, calculate the cosine similarity of each centroid within characters, and build a cosine similarity matrix.
#### 3.1 Word2vec_AMC.ipynb
This file gets character complexity for the AMC show.
#### 3.2 Word2vec_GH.ipynb
This file gets character complexity for the GH show.
#### 3.3 Word2vec_DoOL.ipynb
This file gets character complexity for the DoOL show.

### 4. Character Similarity (Analysis Part 4)
These files contain the between-character similarity (analysis part 4). 
#### 4.1 character_similarity_method1.py
This file runs the 1st method to get character similarity.
#### 4.2 character_similarity_method2.ipynb
This file runs the 2nd method to get character similarity. Also this file calculates the correlations between different between-character similarity measures.
