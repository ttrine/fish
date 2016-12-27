import os, cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

TRAIN_DIR = 'data/train/'
TEST_DIR = 'data/test_stg1/'
FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
ROWS = 90  #720
COLS = 160 #1280
CHANNELS = 3

def get_images(fish):
    """Load files from train folder"""
    fish_dir = TRAIN_DIR+'{}'.format(fish)
    images = [fish+'/'+im for im in os.listdir(fish_dir)]
    #print images
    return images

def read_image(src):
    """Read and resize individual images"""
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (COLS, ROWS), interpolation=cv2.INTER_CUBIC)
    return im

def preprocess_train():
	files = []
	y_all = []

	for fish in FISH_CLASSES:
	    fish_files = get_images(fish)
	    files.extend(fish_files)
	    
	    y_fish = np.tile(fish, len(fish_files))
	    y_all.extend(y_fish)
	    #print("{0} photos of {1}".format(len(fish_files), fish))
	    
	y_all = np.array(y_all)
	print y_all

	X_all = np.ndarray((len(files), ROWS, COLS, CHANNELS), dtype=np.uint8)

	for i, im in enumerate(files): 
	    X_all[i] = read_image(TRAIN_DIR+im)
	    if i%1000 == 0: print('Processed {} of {}'.format(i, len(files)))

	print(X_all.shape)

	# One Hot Encoding Labels
	y_all = LabelEncoder().fit_transform(y_all)
	y_all = np_utils.to_categorical(y_all)

	X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, 
                                                    test_size=0.2, random_state=23, 
                                                    stratify=y_all)
	np.save(TRAIN_DIR+'X_train',X_train)
	np.save(TRAIN_DIR+'X_valid',X_valid)
	np.save(TRAIN_DIR+'y_train',y_train)
	np.save(TRAIN_DIR+'y_valid',y_valid)

def preprocess_test():
	files = [file for file in os.listdir(TEST_DIR) if file != "X_all.npy"]
	files.sort()
	X_all = np.ndarray((len(files), ROWS, COLS, CHANNELS), dtype=np.uint8)

	for i,im in enumerate(files):
		X_all[i] = read_image(TEST_DIR+im)

	print(X_all.shape)

	np.save(TEST_DIR+'X_all',X_all)

if __name__ == '__main__':
	# preprocess_train()
	preprocess_test()