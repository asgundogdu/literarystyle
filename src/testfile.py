#import text_processing #import *
import network
import numpy as np

def main():
	#data = text_processing.data()
	#data.__spacy__()
	#data.__vader_processing__()

	#data.get_processed_data()

	A = np.random.rand(10,10)
	print(A)
	network.similarity_to_edgelist(A)

if __name__ == '__main__':
	main()