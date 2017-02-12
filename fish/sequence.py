import os
import numpy as np
import random
from scipy.ndimage.measurements import label

from keras.callbacks import ModelCheckpoint

def sequencer(chunks,coverage):
	# Identify contiguous regions of coverage map
	labels, numL = label(coverage)
	# Traverse each region, adding each chunk to a list
	coverage_indices = [(labels == i).nonzero() for i in xrange(1, numL+1)]
	location_seq = sum([zip(index[0],index[1]) for index in coverage_indices],[])
	chunk_seq = [chunks[l] for l in location_seq]

	return np.array(chunk_seq), np.array(location_seq)

def random_sequencer(chunks,coverage):
	chunk_seq, location_seq = sequencer(chunks, coverage)
	shuffle = random.sample(range(len(chunk_seq)),len(chunk_seq))
	chunk_seq = chunk_seq[shuffle]
	location_seq = location_seq[shuffle]

	return chunk_seq, location_seq

# Put all nonblack image chunks into a random sequence
def detector_sequencer(chunks, coverage):
	location_seq = []
	for i in range(len(chunks)):
		for j in range(len(chunks[i])):
			if chunks[i,j] is not None:
				location_seq.append((i, j))

	chunk_seq = [chunks[l] for l in location_seq]
	coverage_seq = [coverage[l] for l in location_seq]
	
	chunk_seq = np.array(chunk_seq)
	coverage_seq = np.array(coverage_seq)
	location_seq = np.array(location_seq)

	shuffle = random.sample(range(len(chunk_seq)),len(chunk_seq))
	chunk_seq = chunk_seq[shuffle]
	coverage_seq = coverage_seq[shuffle]
	location_seq = location_seq[shuffle]
	
	return np.array(chunk_seq), np.array(coverage_seq), np.array(location_seq)
