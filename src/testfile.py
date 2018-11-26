try:
	from .text_processing import *
	from .network import *
	from .similarity import *
except:
	from text_processing import *
	from network import *
	from similarity import *

import numpy as np

def main():
	try:
		data = Data()
	except:
		data = Data(directory_path='src/')
	#data.__spacy__()
	#data.__vader_processing__()

	data.get_processed_data()
	data.stem_vocab()

	#A = np.random.rand(10000,10000)
	#print(A)

	sim_matrix = cosine_sim_matrix(data.pdata)
	return sim_matrix

	#network.similarity_to_edgelist(sim_matrix)

if __name__ == '__main__':
	main()