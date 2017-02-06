from keras.models import Sequential
from keras.layers import Dropout, Flatten, ZeroPadding2D, Convolution2D, MaxPooling2D, Dense, BatchNormalization
from keras import backend as K

from fish.detector import DetectorContainer

def construct(n):
	model = Sequential()

	model.add(BatchNormalization(input_shape=(n, n, 3)))
	model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
	model.add(Convolution2D(16, 3, 3, activation='relu', dim_ordering='tf'))
	model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
	model.add(Convolution2D(16, 3, 3, activation='relu', dim_ordering='tf'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))

	model.add(BatchNormalization())
	model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
	model.add(Convolution2D(32, 3, 3, activation='relu', dim_ordering='tf'))
	model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
	model.add(Convolution2D(32, 3, 3, activation='relu', dim_ordering='tf'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))

	model.add(BatchNormalization())
	model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
	model.add(Convolution2D(64, 3, 3, activation='relu', dim_ordering='tf'))
	model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
	model.add(Convolution2D(64, 3, 3, activation='relu', dim_ordering='tf'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))

	model.add(BatchNormalization())
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))

	return model

if __name__ == '__main__':
	import sys # basic arg parsing, infer DetectorContainer name
	name = sys.argv[0].split('/')[-2]

	model = DetectorContainer(name,construct(64),64,"adam")
	model.train(nb_epoch=int(sys.argv[1]), batch_size=int(sys.argv[2]), samples_per_epoch=int(sys.argv[3]))
