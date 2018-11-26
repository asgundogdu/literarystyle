"""
Text processing contains a class that preprocesses the data for similarities by
cleaning strings, tokenizing words, and removing tokens.
"""

import pandas as pd
from time import time
import pickle

# NLP stuff
#import spacy
#import en_core_web_sm
import string
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

from .config import *

class data():

	def __init__(self, directory_path = ''):
		# Whether or not we want to take a subset of the dataframe
		self.subset = config_subset

		if self.subset:
			self.dataframe = pd.read_pickle(directory_path + 'all_the_news.pkl').sample(frac=config_subsample_size).reset_index(drop=True)
		else:
			self.dataframe = pd.read_pickle(directory_path + 'all_the_news.pkl')

		#self.nlp = en_core_web_sm.load()

		self.stopwords = stopwords.words('english')#{s : True for s in stopwords.words('english')}
		self.stemmer = SnowballStemmer('english')
		self.lmtzr = WordNetLemmatizer()

	# Will process all the text in spacy if we want to. Lot's of different nlp options. Note, rate is about 5000
	# articles per hour on 16gb RAM
	def __spacy__(self):

		if self.subset:
			self.dataframe = self.dataframe.sample(frac=config_subsample_size).reset_index(drop=True)

		start = time()
		self.spacy_text = {}
		for idx, row in self.dataframe.iterrows():
			self.spacy_text[idx] =  self.nlp(row['content'])

			if not idx % 5000:
				print(idx, "rows in", (time()-start)/60, 'min')

	def vader_polarity(self):
		self.vader_analyzer = SentimentIntensityAnalyzer()
		self.vader_list = []

		start = time()
		tc = 0
		projlen = len(self.pdata)
		for t in self.pdata: 
		    #self.vader_list.append([self.vader_analyzer.polarity_scores(i) for i in t.split()])
		    aplist = []
		    for w in t:
		    	ap = self.vader_analyzer.polarity_scores(w)
		    	ap['word'] = w
		    	aplist.append(ap)
		    
		    self.vader_list.append(aplist)
		    
		    if tc == 100:
		        amount = projlen
		        tproj = (time()-start) * (amount/100) / 60
		        print("Projected time to " + str(amount) + ":", tproj, 'min')
		    if not tc % 10000:
		        print(tc,'in', (time()-start)/60, 'min')
		    tc+=1

		with open('vader_list_wo_stpwrds.pkl', 'wb') as outfile:
			pickle.dump(self.vader_list, outfile)

	def filter_by_sentiment(self):
		self.pdata = [[i['word'] for i in j if abs(i['compound']) < config_sent_threshold] for j in self.vader_list]

	def stem_vocab(self, w_lemma=True):
		start = time()

		if w_lemma:
			self.pdata = [[self.lmtzr.lemmatize(self.stemmer.stem(word)) for word in j] for j in self.pdata]
			print('Stemming and lemmatization done in', (time()-start) / 60, 'min')
		else:
			self.pdata = [[self.stemmer.stem(word) for word in j] for j in self.pdata]
			print('Stemming done in', (time()-start) / 60, 'min')

	# The primary function that builds the processed data
	# Once run, the data that can be output is self.pdata
	# Could also just return pdata later on, if that's a better design choice
	def get_processed_data(self, load = False):
		if load:
			# Placeholder
			pass
		else:
			self.pdata = self.dataframe.content.tolist()

			start = time()

			self.pdata = [[i for i in re.sub(r'[^\w\s]','',c.lower()).split() if i not in self.stopwords] for c in self.pdata]
			# Numerical processing from homework assignment 4
			self.pdata = [['NUM' if re.match('[0-9]+', word) is not None else word for word in c ] for c in self.pdata]
			print('Simple splitting done in:', (time()-start)/60, 'min')

