import numpy as np
import h5py, pandas

from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

from fish.chunk import chunk_mask, chunk_image
from fish.sequence import train_sequencer

data = h5py.File('data/train/binary/data.h5','r')

X_test = data['X_test']
y_masks_test = data['y_masks_test']

def compute_and_write(n):
	chunk_sequences = []
	location_sequences = []
	coverage_sequences = []
	print "Generating sequences..."
	for i in range(len(X_test)):
		chunk_matrix = chunk_image(n,X_test[i])
		coverage_matrix = chunk_mask(n,chunk_matrix,y_masks_test[i])

		chunk_sequence, coverage_sequence, location_sequence = train_sequencer(chunk_matrix, coverage_matrix)
		
		chunk_sequences.append(chunk_sequence)
		coverage_sequences.append(coverage_sequence)
		location_sequences.append(location_sequence)

	print "Padding sequences (takes a while)..."
	chunk_sequences = pad_sequences(chunk_sequences)
	coverage_sequences = pad_sequences(coverage_sequences)
	location_sequences = pad_sequences(location_sequences)

	print "Saving arrays..."
	np.save("data/train/binary/X_test_chunk_seqs_"+str(n),chunk_sequences)
	np.save("data/train/binary/y_test_coverage_seqs_"+str(n),coverage_sequences)
	np.save("data/train/binary/X_test_loc_seqs_"+str(n),location_sequences)

if __name__ == '__main__':
	import sys
compute_and_write(int(sys.argv[1]))
