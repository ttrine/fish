import numpy as np
import h5py, pandas

from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

from fish.chunk import chunk_mask, chunk_image
from fish.sequence import detector_sequencer

data = h5py.File('data/train/binary/data.h5','r')

X_test = data['X_test']
y_masks_test = data['y_masks_test']

def compute_and_write(n):
	chunk_sequences = []
	location_sequences = []
	coverage_matrices = []
	print "Generating sequences..."
	for i in range(len(X_test)):
		chunk_matrix = chunk_image(n,X_test[i])
		coverage_matrix = chunk_mask(n,chunk_matrix,y_masks_test[i])
		coverage_matrices.append(coverage_matrix)

		chunk_sequence, location_sequence = detector_sequencer(chunk_matrix, coverage_matrix)
		
		chunk_sequences.append(chunk_sequence)
		location_sequences.append(location_sequence)

	print "Padding sequences (takes a while)..."
	chunk_sequences = pad_sequences(chunk_sequences)
	location_sequences = pad_sequences(location_sequences)

	print chunk_sequences.shape
	print coverage_matrices.shape
	print location_sequences.shape
	print class_labels.shape
	np.save("data/train/binary/X_test_det_chunk_seqs_"+str(n),chunk_sequences)
	np.save("data/train/binary/y_test_det_coverage_mats_"+str(n),coverage_matrices)
	np.save("data/train/binary/X_test_det_loc_seqs_"+str(n),location_sequences)

if __name__ == '__main__':
	import sys
	compute_and_write(int(sys.argv[1]))
