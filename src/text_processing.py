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
from nltk.corpus import stopwords

from .config import *

class data():

	def __init__(self, directory_path = ''):
		self.dataframe = pd.read_pickle(directory_path + 'all_the_news.pkl')

		#self.nlp = en_core_web_sm.load()

		self.stopwords = stopwords.words('english')#{s : True for s in stopwords.words('english')}

		# Whether or not we want to take a subset of the dataframe
		self.subset = True

	# Will process all the text in spacy if we want to. Lot's of different nlp options. Note, rate is about 5000
	# articles per hour on 16gb RAM
	def __spacy__(self):

		if self.subset:
			self.dataframe = self.dataframe.sample(frac=subsample_size).reset_index(drop=True)

		start = time()
		self.spacy_text = {}
		for idx, row in self.dataframe.iterrows():
			self.spacy_text[idx] =  self.nlp(row['content'])

			if not idx % 5000:
				print(idx, "rows in", (time()-start)/60, 'min')

	def __vader_polarity__(self):
		self.vader_list = []

		start = time()
		tc = 0
		for t in temp: 
		    self.vader_list.append([a.polarity_scores(i) for i in t.split()])
		    
		    if tc == 100:
		        amount = 150000
		        tproj = (time()-start) * (amount/100) / 60
		        print("Projected time to" + str(amount) + ":", tproj, 'min')
		    if not tc % 10000:
		        print(tc,'in', (time()-start)/60, 'min')
		    tc+=1

		with open('vader_list.pkl', 'wb') as outfile:
			pickle.dump(self.vader_list, outfile)

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

			self.pdata = [[i for i in c.lower().split() if i not in self.stopwords] for c in self.pdata]
			print('Simple splitting done in:', (time()-start)/60, 'min')

