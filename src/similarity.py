"""
Computes document similarities
"""

from sklearn.feature_extraction.text import TfidfVectorizer

def cosine_sim_matrix(documents):
	"""
	Can recieve list of lists or list of strings
	Outputs a nxn similarity matrix
	"""

	# As of now, documents are to come in as list of words. But want to stay flexible
	if not isinstance(documents[0], str):
		documents = [" ".join(doc) for doc in documents]

	tfidf = TfidfVectorizer().fit_transform(documents)

	pairwise_sim = tfidf * tfidf.T

	return pairwise_sim