from time import time
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

class Sent():

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

	def filter_by_sentiment(self, threshold = None):
		if threshold is None:
			threshold = config_sent_threshold

		self.pdata = [[i['word'] for i in j if abs(i['compound']) < threshold] for j in self.vader_list]

	def filter_neutral_sent(self):
		self.pdata = [[i['word'] for i in j if abs(i['compound']) != 0] for j in self.vader_list]