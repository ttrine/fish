import os
import h5py
import numpy as np
import cv2

IMG_DIR = 'data/test_stg1/raw/'
filenames = os.listdir(IMG_DIR)

# Put each image in the top-left of a tensor
X_eval = np.zeros((1000,974,1732,3))
for i, filename in enumerate(filenames):
	img = cv2.imread(IMG_DIR + filename)
	X_eval[i,0:img.shape[0],0:img.shape[1]] = img

# Create hdf5 dataset for X_eval
eval_data = h5py.File("data/test_stg1/binary/eval_data.h5",'w')
X_eval = eval_data.create_dataset('X_eval', data=X_eval)

# Save out filenames tensor too
np.save("data/test_stg1/binary/y_filenames.npy",filenames)
