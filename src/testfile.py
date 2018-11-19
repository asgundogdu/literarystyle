import text_processing #import *

def main():
	data = text_processing.data()
	#data.__spacy__()
	data.__vader_processing__()

	data.get_processed_data()

if __name__ == '__main__':
	main()