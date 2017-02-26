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
	coverage_matrices = []
	for i in range(len(X_test)):
		chunk_matrix = chunk_image(n,X_test[i])
		coverage_matrices.append(chunk_mask(n,chunk_matrix,y_masks_test[i]))

	coverage_matrices = np.array(coverage_matrices)
	y_classes_test = np.load('data/train/binary/y_classes_test.npy')
	y_classes_test = np_utils.to_categorical(pandas.factorize(y_classes_test, sort=True)[0])

	print "Saving arrays..."
	np.save("data/train/binary/y_test_coverage_mats_"+str(n),coverage_matrices)
	np.save("data/train/binary/y_classes_test_onehot",y_classes_test)

if __name__ == '__main__':
	import sys
	compute_and_write(int(sys.argv[1]))
