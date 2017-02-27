from keras.models import Model

from keras.layers import Input, Dense, Flatten, merge, Reshape, ZeroPadding2D, Convolution2D, MaxPooling2D, BatchNormalization
from keras.layers.core import Masking, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM

from fish.detect import DetectorContainer

def construct():
	imgs = Input(shape=(487, 866, 3))
	batch = BatchNormalization()(imgs)

	# Shared CNN for learning representations common to detection and classification.
	conv1 = ZeroPadding2D((3, 3))(batch)
	conv1 = Convolution2D(16, 5, 5, activation='relu')(conv1)
	conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = ZeroPadding2D((3, 3))(conv1)
	conv2 = Convolution2D(32, 5, 5, activation='relu')(conv2)
	conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = ZeroPadding2D((3, 3))(conv2)
	conv3 = Convolution2D(64, 5, 5, activation='relu')(conv3)
	conv3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = ZeroPadding2D((3, 3))(conv3)
	conv4 = Convolution2D(128, 5, 5, activation='relu')(conv4)
	conv4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = ZeroPadding2D((1, 1))(conv4)
	conv5 = Convolution2D(256, 3, 3, activation='relu')(conv5)
	conv5 = ZeroPadding2D((1, 1))(conv5)
	conv5 = Convolution2D(1, 3, 3, activation='sigmoid')(conv5)
	conv5 = MaxPooling2D(pool_size=(2, 2))(conv5)

	# Shave off channel dimension
	pred_mat = Reshape((16,28))(conv5)

	return Model(input=imgs,output=pred_mat)

if __name__ == '__main__':
	import sys # basic arg parsing, infer name
	name = sys.argv[0].split('/')[-2]
	
	if len(sys.argv) < 4:
		print "Usage: train nb_epoch batch_size samples_per_epoch"
		sys.exit()

	model = DetectorContainer(name,construct(),32,"adam")
	model.train(nb_epoch=int(sys.argv[1]), batch_size=int(sys.argv[2]), samples_per_epoch=int(sys.argv[3]))
