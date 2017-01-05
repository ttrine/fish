import os, json, cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

TRAIN_DIR = 'data/train/'
TEST_DIR = 'data/test_stg1/'
FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
ROWS = 256  #720
COLS = 256 #1280
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

def preprocess_masks():
	files = []

	# Only process images with fish
	FISH_CLASSES.remove('NoF')

	for fish in FISH_CLASSES:
	    fish_files = get_images(fish)
	    files.extend(fish_files)

	annotations = {}
	annotations['ALB'] = json.load(open('data/train/bounding_box/alb_labels.json'))
	annotations['BET'] = json.load(open('data/train/bounding_box/bet_labels.json'))
	annotations['DOL'] = json.load(open('data/train/bounding_box/dol_labels.json'))
	annotations['LAG'] = json.load(open('data/train/bounding_box/lag_labels.json'))
	annotations['OTHER'] = json.load(open('data/train/bounding_box/other_labels.json'))
	annotations['SHARK'] = json.load(open('data/train/bounding_box/shark_labels.json'))
	annotations['YFT'] = json.load(open('data/train/bounding_box/yft_labels.json'))

	X_all = np.ndarray((len(files), ROWS, COLS, CHANNELS), dtype=np.uint8)
	y_all = np.ndarray((len(files), ROWS, COLS), dtype=np.uint8)

	for i, im in enumerate(files):
		fish_class = im.split('/')[0]
		filename = im.split('/')[1]

		# Get bounding box annotation, handling a formatting idiosyncrasy
		if fish_class == 'OTHER' or fish_class == 'SHARK' or fish_class == 'YFT':
			record = [x for x in annotations[fish_class] if x['filename'].split('/')[-1]==filename][0]
		else:
			record = [x for x in annotations[fish_class] if x['filename']==filename][0]
		
		# Create mask of same size as original image, then resize both
		im = cv2.imread(TRAIN_DIR+im, cv2.IMREAD_COLOR)
		WIDTH = len(im[0,])
		HEIGHT = len(im)
		mask = np.zeros((HEIGHT,WIDTH))
		for annotation in record['annotations']:
			x = int(round(annotation['x']))
			width = int(round(annotation['width']))
			y = int(round(annotation['y']))
			height = int(round(annotation['height']))
			if x < 0: x = 0
			if y < 0: y = 0
			if x + width >= WIDTH: width = WIDTH - 1 - x
			if y + height >= HEIGHT: height = HEIGHT - 1 - y
			for j in range(x, x+width):
				for k in range(y,y+height):
					mask[k,j]=1.
		im = cv2.resize(im, (COLS, ROWS), interpolation=cv2.INTER_CUBIC)
		mask = cv2.resize(mask, (COLS, ROWS), interpolation=cv2.INTER_NEAREST)
		if i%1000 == 0: print('Processed {} of {}'.format(i, len(files)))
		
		X_all[i] = im
		y_all[i] = mask

	X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, 
                                                    test_size=0.2, random_state=23, 
                                                    stratify=y_all)
	np.save(TRAIN_DIR+'X_train_localizer',X_train)
	np.save(TRAIN_DIR+'X_valid_localizer',X_valid)
	np.save(TRAIN_DIR+'y_train_localizer',y_train)
	np.save(TRAIN_DIR+'y_valid_localizer',y_valid)

if __name__ == '__main__':
	preprocess_train()
	preprocess_test()
	preprocess_masks()