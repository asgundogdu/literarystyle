"""
Network.py contstucts a network based on different similarity thresholds.
	Input: similaties (nxn similarity matrix)
	Output: edgelist
"""

# Need relative import if in higher directory, but will through error if running from same directory
try:
	from .config import *
except:
	from config import *

def similarity_to_edgelist(sim_matrix):
	"""
	Should take the output of standard doc2vec similarities from gensim
	"""

	edgelist = []
	place = 0
	for i, m in enumerate(sim_matrix):
		for j, n in enumerate(m):
			# use placeholder to avoid repeated links
			if j < place:
				continue
			elif n > config_sim_threshold:
				edgelist.append([i,j])

		place += 1

	return edgelist