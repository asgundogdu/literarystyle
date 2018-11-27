# literarystyle
Understanding Sentiment Effect On Similarity Networks for All the News - UML Course Project

## Project Overview

The objective of this project is to understand how presence or absense of emotionally charged words effect communities of news documents. The project flow is outlined below:

1. Clean, tokenize, stem, and lemmatize news articles
2. Calculate document similarities
3. Create network of similarities using similarity threshold
4. Run community detection on similarity network
5. Remove words that exceed a sentiment intensity threshold and repeat steps 1-4
6. Evaluate the changes between similarity networks

## Milestone 1

Please see the notebooks where we perform data cleaning, outlier detection and checked data distributions.

`data_exploration.ipynb`

`Data Exploration 2.ipynb`

## Milestone 2

- Major aspects of data pipeline completed in src folder
- More data exploration for sentiment in data_exploration.ipynb
- Beginning to experiment notebook which will direct the output of the project