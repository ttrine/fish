import os
import numpy as np
import random

# Put all nonblack image chunks into a random sequence
def train_sequencer(chunks, coverage):
	location_seq = []
	for i in range(len(chunks)):
		for j in range(len(chunks[i])):
			if chunks[i,j] is not None:
				location_seq.append((i, j))

	chunk_seq = [chunks[l] for l in location_seq]
	coverage_seq = [coverage[l] for l in location_seq]
	
	chunk_seq = np.array(chunk_seq)
	location_seq = np.array(location_seq)
	coverage_seq = np.array(coverage_seq)

	shuffle = random.sample(range(len(chunk_seq)),len(chunk_seq))
	chunk_seq = chunk_seq[shuffle]
	coverage_seq = coverage_seq[shuffle]
	location_seq = location_seq[shuffle]
	
	return np.array(chunk_seq), np.array(coverage_seq), np.array(location_seq)

def eval_sequencer(chunks):
	location_seq = []
	for i in range(len(chunks)):
		for j in range(len(chunks[i])):
			if chunks[i,j] is not None:
				location_seq.append((i, j))
	
	chunk_seq = [chunks[l] for l in location_seq]
	
	chunk_seq = np.array(chunk_seq)
	location_seq = np.array(location_seq)
	
	shuffle = random.sample(range(len(chunk_seq)),len(chunk_seq))
	chunk_seq = chunk_seq[shuffle]
	location_seq = location_seq[shuffle]

	return np.array(chunk_seq), np.array(location_seq)
