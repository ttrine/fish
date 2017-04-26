import os
import h5py
import numpy as np
import cv2

def make_eval_data():
	IMG_DIR = 'data/test_stg1/raw/'
	filenames = os.listdir(IMG_DIR)

	# Put each image in the top-left of a tensor
	X_eval = np.zeros((1000,974/2,1732/2,3), dtype=np.uint8)
	for i, filename in enumerate(filenames):
		img = cv2.imread(IMG_DIR + filename)
		img = cv2.resize(img,(1732/2,974/2))
		X_eval[i,0:img.shape[0],0:img.shape[1]] = img

	# Create hdf5 dataset for X_eval
	eval_data = h5py.File("data/test_stg1/binary/eval_data.h5",'w')
	X_eval = eval_data.create_dataset('X_eval', data=X_eval)

	# Save out filenames tensor too
	np.save("data/test_stg1/binary/y_filenames.npy",filenames)

if __name__ == '__main__':
	make_eval_data()
