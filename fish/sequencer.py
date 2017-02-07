import numpy as np
from scipy.ndimage.measurements import label

def sequence(chunks,coverage):
	# Identify contiguous regions of coverage map
	labels, numL = label(coverage)
	# Traverse each region, adding each chunk to a list
	coverage_indices = [(labels == i).nonzero() for i in xrange(1, numL+1)]
	location_seq = sum([zip(index[0],index[1]) for index in coverage_indices],[])
	chunk_seq = [chunks[l] for l in location_seq]

	return np.array(chunk_seq), np.array(location_seq)
