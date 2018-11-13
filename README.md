# literarystyle
Understanding Literary Style within the News Corpus - UML Course Project

## Data Explorations

Please see the notebooks where we perform data cleaning, outlier detection and checked data distributions. 

`data_exploration.ipynb`

`Data Exploration 2.ipynb`

Based on the tf-idf analysis and PCA visualizations, we say that this challenging problem of clustering news source or authors from the text regardless of the topic is challenging.

For future implementations, we envision to use LSTM networks for getting document representation (by optimizing triplet loss), then plan to create (dis)similarity network/graph where we can trim the edges based on some threshold (CV) and then apply community detection algorithm.

We can expect the resulting clusters will be something like the article source/author clusters as we extract our representations by optimizing triplet loss for either article source or authors.

The worst case we can try to understand - what are the clusters or communities we will found on the network, are refering to...
