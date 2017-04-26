import numpy as np
import h5py, pandas
import cv2

from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

from fish.chunk import chunk_mask, chunk_image
from fish.sequence import train_sequencer

data = h5py.File('data/train/binary/data.h5','r')

X_test = data['X_test']
y_masks_test = data['y_masks_test']

def compute_and_write(n, resize_factor):
	nrow = 974 / resize_factor
	ncol = 1732 / resize_factor

	images = []
	coverage_matrices = []
	for i in range(len(X_test)):
		image = cv2.resize(X_test[i], (ncol, nrow))
		mask = cv2.resize(y_masks_test[i], (ncol, nrow), interpolation = cv2.INTER_NEAREST)
		images.append(image)
		chunk_matrix = chunk_image(n,image,resize_factor)
		coverage_matrices.append(chunk_mask(n,chunk_matrix,mask,resize_factor))

	images = np.array(images)
	coverage_matrices = np.array(coverage_matrices)
	y_classes_test = np.load('data/train/binary/y_classes_test.npy')
	y_classes_test = np_utils.to_categorical(pandas.factorize(y_classes_test, sort=True)[0])

	print "Saving arrays..."
	np.save("data/train/binary/X_test_"+str(resize_factor),images)
	np.save("data/train/binary/y_test_coverage_mats_"+str(n),coverage_matrices)
	np.save("data/train/binary/y_classes_test_onehot",y_classes_test)

if __name__ == '__main__':
	import sys
	compute_and_write(int(sys.argv[1]), int(sys.argv[2]))
