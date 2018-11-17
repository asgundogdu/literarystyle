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

	def __spacy__(self):

		if self.subset:
			self.dataframe = self.dataframe.sample(frac=config.subsample_size).reset_index(drop=True)

		start = time()
		self.spacy_text = {}
		for idx, row in self.dataframe.iterrows():
			self.spacy_text[idx] =  self.nlp(row['content'])

			if not idx % 5000:
				print(idx, "rows in", (time()-start)/60, 'min')

