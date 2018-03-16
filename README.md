# Computational Content Analysis Project: Gender Roles in American Soap Operas

Team members: Hyunku Kwon(MAPSS), Oscar Stuhler(MAPSS), Xingyun Wu(MACSS)

This is the GitHub repository of our final project of the Computational Content Analysis class. With the transcripts of All My Children, General Hospital, and Days of Our Lives, we analyze gender roles in American Soap Operas.

This is the README file of our annotated scripts.

## Part I: Data Scraping

### get_url.py	Web Scraper to Get All the Urls
	Transcripts of each episode are stored in seperated webpages. We use this file to build a scraper to get all the urls of the webpages. And export the urls to a csv file.

### get_text.py	Text Scraper
	This file is to scrape the transcripts of soap operas from webpages. It uses the url.csv file constructed in 'get_url.py'. And it stores the scraped texts in .txt files by year and by show.


## Part II: Data Processing

### parse_text.py	Parse the Plain Texts
	This file takes the raw .txt files of data scraped from the internet, and get them into the format ready for analysis.


## Part III: Data Analysis

