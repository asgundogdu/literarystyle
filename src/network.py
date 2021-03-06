"""
Network.py contstucts a network based on different similarity thresholds.
	Input: similaties (nxn similarity matrix)
	Output: edgelist
"""

from time import time

# Need relative import if in higher directory, but will through error if running from same directory
try:
	from .config import *
except:
	from config import *

def similarity_to_edgelist(sim_matrix, labels = None, custom_sim_threshold = None):
	"""
	Should take the output of standard doc2vec similarities from gensim
	"""

	if custom_sim_threshold is not None:
		sim_threshold = custom_sim_threshold
	else:
		sim_threshold = config_sim_threshold

	if labels is None:
		labels = list(range(len(sim_matrix)))

	edgelist = []
	place = 0
	for i, m in enumerate(sim_matrix):
		for j, n in enumerate(m):
			# use placeholder to avoid repeated links
			#print(n)
			if j < place:
				continue
			elif i == j:
				continue	# Don't want to include self loops
			elif n > sim_threshold:
				edgelist.append([labels[i],labels[j]])

		place += 1

	return edgelist

# Function designed to tune the similarity threshold until the desired percentage of links are achieved
def tune_sim_thresh(sim_matrix, perc_links):
	num_nodes = len(sim_matrix)
	num_possible_links = (num_nodes * (num_nodes-1)) / 2
	target_num_links = num_possible_links * perc_links

	target_upper_bound = target_num_links * 1.3
	target_lower_bound = target_num_links * .7

	num_links = 0
	threshold = config_sim_threshold	# Initialize threshold based on config settings
	num_loops = 0
	smoothing = 2
	start = time()
	while True:
		edgelist = similarity_to_edgelist(sim_matrix, custom_sim_threshold = threshold)
		num_links = len(edgelist)
		
		if ((target_lower_bound <= num_links) & (num_links <= target_upper_bound)): break
		elif num_loops == 100: break

		# Will only be two cases that num_links doesn't fall in interval
		# A smarter tune step is probably possible
		if num_links < target_lower_bound:
			decrease = 1 - (num_links / target_num_links)

			threshold = threshold - (threshold * decrease) / smoothing
			print('decrease:', decrease)
		else: 
			increase = 1 - (target_num_links / num_links)
			threshold = threshold + (threshold * increase) / smoothing
			print('increase:', increase)

		#if not num_loops % 10:
		print(num_loops, 'loops in', (time() - start)/60, 'min')
		print('Upper bound:', target_upper_bound/num_possible_links,
		'   Lower bound:', target_lower_bound/num_possible_links,
		'   Current:', num_links/num_possible_links)
		print(num_loops, 'loops in', (time() - start)/60, 'min')
		print('Upper bound:', target_upper_bound,
		'   Lower bound:', target_lower_bound,
		'   Current:', num_links)

		num_loops += 1
		smoothing += 1

	print('Number of loops to converge:', num_loops)
	print('Converged in', (time() - start)/60, 'min')
	print('Similarity threshold after tuning:', threshold)

	return edgelist