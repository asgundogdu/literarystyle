"""
Text processing contains a class that preprocesses the data for similarities by
cleaning strings, tokenizing words, and removing tokens.
"""

import pandas as pd
import numpy as np
from time import time
import pickle
from collections import Counter

# NLP stuff
#import spacy
#import en_core_web_sm
import string
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# Need relative import if in higher directory, but will through error if running from same directory
try:
	from .config import *
	from .sentiment import Sent
except:
	from config import *
	from sentiment import Sent

class Data(Sent):

	def __init__(self, directory_path = '', process_code = 0, outpath = ''):
		# Whether or not we want to take a subset of the dataframe
		self.subset = config_subset
		self.process_code = process_code
		self.outpath = outpath

		self.dataframe = pd.read_pickle(directory_path + 'all_the_news.pkl')

		# if self.subset:
		# 	self.dataframe = pd.read_pickle(directory_path + 'all_the_news.pkl').sample(frac=config_subsample_size)
		# 	self.labels = self.dataframe.index.tolist()
		# 	self.dataframe = self.dataframe.reset_index(drop=True)
		# else:
		# 	self.dataframe = pd.read_pickle(directory_path + 'all_the_news.pkl')
		# 	self.labels = self.dataframe.index.tolist()

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
	def get_processed_data(self, author_threshold = 10, load = False):
		if load:
			# Placeholder
			pass
		else:
			article_df = self.dataframe

			# Data cleaning
			article_df = article_df[~article_df.author.isna() & ~article_df.title.isna()]

			dct = dict(Counter(article_df.author.tolist()))
			filtered_users = [key for key in dct.keys() if dct[key]>author_threshold]
			article_df = article_df[article_df.publication.isin(filtered_users)]

			if self.subset:
				article_df = article_df.sample(frac=config_subsample_size)

			article_df = article_df.set_index('id', drop=True)
			self.labels = article_df.index.tolist()
			article_df = article_df.reset_index()
			self.metadata = article_df.reset_index()[['id', 'title', 'publication', 'author', 'date', 'year', 'month']].copy().reset_index()

			self.metadata.id = self.metadata.id.fillna(-1).astype('int64')
			self.metadata.month =  self.metadata.month.fillna(-1).astype('int64')
			self.metadata.year = self.metadata.year.fillna(-1).astype('int64')

			label_output = pd.DataFrame({'id' : self.labels}).reset_index()

			if config_write_labels:
				label_output.to_csv(self.outpath + 'label_mapping_' + str(self.process_code) + '.csv', index=False)
				self.metadata.to_csv(self.outpath + 'metadata_by_mapping_' + str(self.process_code) + '.csv', index=False)

			self.pdata = article_df.content.tolist()

			start = time()

			self.pdata = [[i for i in re.sub(r'[^\w\s]','',c.lower()).split() if i not in self.stopwords] for c in self.pdata]
			# Numerical processing from homework assignment 4
			self.pdata = [['NUM' if re.match('[0-9]+', word) is not None else word for word in c ] for c in self.pdata]
			print('Simple splitting done in:', (time()-start)/60, 'min')

