"""
Computes document similarities
"""
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse.csgraph import minimum_spanning_tree as mst_nsim
import numpy as np
import pandas as pd


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


def _mst_sym(A, return_LongestLinks=True):
    """scipy mst (kruskal) return triangular matrix as mst"""
    dim = A.shape[0]
    mst = mst_nsim(A).todense()
    mst[mst > 0] = 1
    remained_edges = np.maximum(mst, mst.T)

    return remained_edges


def _knn_mst(D, k=13): # BUG WORKS WITH DISTANCE MATRIX (+++++++++++++++++)\
    """Spectral clustering is a technique that uses the spectrum of a similarity graph to cluster data. 
    Part of this procedure involves calculating the similarity between data points and creating a similarity graph from the resulting similarity matrix. 
    This is ordinarily achieved by creating a k-nearest neighbour (kNN) graph
    Source: https://kclpure.kcl.ac.uk/portal/en/publications/spectral-clustering-using-the-knnmst-similarity-graph(d1d35174-4ced-4b78-84a7-5e3e6d359e82).html"""
    n = D.shape[0]
    assert (D.shape[0] == D.shape[1])

    np.fill_diagonal(D, 0)
    A = np.zeros((n, n))
    for i in range(n):
        ix = np.argsort(D[i, :])
        A[i, ix[1]] = 1  # Connect to the nearest node after itself
        A[ix[1], i] = 1  # The same on other direction

        for j in range(k - 1):
            ij = j + 1
            #             if D[i, ix[ij]] < theta:
            A[i, ix[ij]] = 1
            A[ix[ij], i] = 1

    mst_remained_edges = _mst_sym(D, False)
    remained_edges = np.maximum(A, mst_remained_edges)

    return remained_edges


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {}:".format(topic_idx))
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


def prune_fully_connected(D, k, verbose):
    pruned_network = _knn_mst(D, k)
    if verbose: print("After pruning network has {} edges".format(np.count_nonzero(pruned_network)))
    return pruned_network


def symm_KL_sim_matrix(documents, LDA_topic_num = 16, mnitsknn_k = 13, prune = True, feature_extractor = 'tfidf', stop_words = True, verbose=True):
    if not stop_words: 
        stop = []
        if verbose: print('Stop words are not removed!')
    else: stop = stopwords.words('english')

    # As of now, documents are to come in as list of words. But want to stay flexible
    if not isinstance(documents[0], str): documents = [" ".join(doc) for doc in documents]

    if verbose: print('Computing word features using {}...'.format(feature_extractor))
    if feature_extractor == 'tfidf':
        bow_transformer = TfidfVectorizer(use_idf=True, smooth_idf=True, ngram_range=(1,2), stop_words=stop, min_df=0.09, max_df=0.91)
        tf_vec = bow_transformer.fit_transform(documents.content.tolist())
    elif feature_extractor == 'count_vec':
        bow_transformer = CountVectorizer(max_df=0.91, min_df=.09,  stop_words=stop)
        tf_vec = bow_transformer.fit_transform(documents.content.tolist())

    tf_feature_names = bow_transformer.get_feature_names()

    n_topics = LDA_topic_num
    print('\nCalculating LDA for {} topics...'.format(n_topics))
    lda = LatentDirichletAllocation(n_components=n_topics, 
                                    max_iter=7, verbose=int(verbose),
                                    learning_method='batch', batch_size=512,
                                    random_state=1).fit(tf_vec)

    if verbose: display_topics(lda, tf_feature_names, no_top_words=10)

    theta = lda.transform(tf_vec)

    if verbose: print('\nTheta has been created with dimensions {}'.format(theta.shape))

    if verbose: print('\nCalculating log_odds from theta matrix... Might take a while')
    
    # Symmetric KL Divergence Calculation
    log_odds = np.log(theta[:,None,:]) - np.log(theta[None,:,:])

    if verbose: print('\nLog odds has been created with dimensions {}'.format(log_odds.shape))

    if verbose: print('\nCalculating Symmetric KL Divergence values of pairs... Takes a while')
    
    sym_kl = 0.5 * ((theta[:,None,:]*log_odds).sum(-1) - (theta[None,:,:] * log_odds).sum(-1))
    
    if verbose: print('\nSymmetric KL Divergence values are calculated and dimensions are {}'.format(sym_kl.shape))

    if prune: pairwise_sim = prune_fully_connected(sym_kl, mnitsknn_k, verbose)

    return pairwise_sim