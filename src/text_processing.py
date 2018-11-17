"""
Text processing contains a class that preprocesses the data for similarities by
cleaning strings, tokenizing words, and removing tokens.
"""

import pandas as pd
from time import time

# NLP stuff
import spacy
import en_core_web_sm

import config

class data():

	def __init__(self):
		self.dataframe = pd.read_pickle('all_the_news.pkl')

		self.nlp = en_core_web_sm.load()

		# Whether or not we want to take a subset of the dataframe
		self.subset = True

	# Will process all the text in spacy if we want to. Lot's of different nlp options. Note, rate is about 5000
	# articles per hour on 16gb RAM
	def __spacy__(self):

		if self.subset:
			self.dataframe = self.dataframe.sample(frac=config.subsample_size).reset_index(drop=True)

		start = time()
		self.spacy_text = {}
		for idx, row in self.dataframe.iterrows():
			self.spacy_text[idx] =  self.nlp(row['content'])

			if not idx % 5000:
				print(idx, "rows in", (time()-start)/60, 'min')

	# The primary function that builds the processed data
	# Once run, the data that can be output is self.pdata
	# Could also just return pdata later on, if that's a better design choice
	def get_processed_data(self):

		self.pdata = self.dataframe.content.tolist()

		start = time()

		self.pdata = [c.split() for c in self.pdata]

		print('Simple splitting done in:', (time()-start)/60, 'min')

