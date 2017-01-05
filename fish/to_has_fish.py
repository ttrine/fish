import numpy as np

def to_has_fish(y):
	nof = np.array([1. if x==1 else 0. for x in y[:,4]])
	f = (~nof.astype(bool)).astype('float64')
	y2 = np.zeros((756,2),'float64')
	y2[:,0] = f
	y2[:,1] = nof
	return y2

if __name__ == '__main__':
	y_valid = np.load('data/train/y_valid.npy')
	y_valid_has_fish = to_has_fish(y_valid)

	y_train = np.load('data/train/y_train.npy')
	y_train_has_fish = to_has_fish(y_train)
	# probably want to write them out to disk too
